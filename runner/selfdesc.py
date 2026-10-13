"""Forensic self-descriptions on the server, then the temporal tests. Unattended and resumable.

    python -m runner.selfdesc --manifest m.csv --out-dir experiments/selfdesc-36k --smoke
    python -m runner.selfdesc --manifest m.csv --out-dir experiments/selfdesc-36k
    python -m runner.selfdesc --manifest arms.csv --out-dir experiments/selfdesc-arms-native \
        --configs K8_M11_B11_L3 K8_M11_B11_L3_rgb --channels luma rgb \
        --temporal-configs K8_M11_B11_L3 K8_M11_B11_L3_rgb --suite arms

Normally started through `./runner/run.sh selfdesc start ...`. The science lives in main's
selfdesc/ (DECISIONS.md D-006..D-009); this module only orchestrates it, the way pipeline.py
orchestrates main's detectors/:

  1 prepare   `selfdesc.sweep --frozen --plan-json`: writes images.parquet, checks that every
              FRE is the one frozen on the local sample (fetching the authors' released one if
              absent), and lists the configs. --frozen makes training a new FRE an error.
  2 extract   one chunk = one shard of one config = one `selfdesc.extract --shard S` subprocess
              in main's venv, one BLAS thread, under nice. K8_M11_B11_L3 first (the confirmatory
              test needs only it), then the rest cheapest-first.
  3 assemble  `selfdesc.extract --assemble` for each config whose shards are all on disk.
  4 temporal  `selfdesc.temporal --suite S` for each of --temporal-configs, over the assembled
              descriptions. It caches every block, so a stop or a crash resumes rather than
              restarting the multi-hour permutation tests.

Several such runs back to back: runner/selfdesc_queue.py (`./runner/run.sh selfdesc queue`).

Interrupting is safe at any point, for the same reasons as the panel run: a shard file that
exists is complete (written to a temp name and renamed), anything left active goes back to
pending, and reconcile() rebuilds chunk status from the shards and assembled configs on disk.
"""

import argparse
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import state as state_module  # noqa: E402


def sha256_file(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(block), b""):
            digest.update(piece)
    return digest.hexdigest()


def child_env():
    env = dict(os.environ)
    env["AIGENBENCH_DATA_ROOT"] = config.DATA_ROOT
    # Each subprocess is one worker. Letting numpy's BLAS spawn 48 threads inside each of 40
    # workers would oversubscribe the box ~40x.
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS"):
        env[var] = "1"
    return env


def main_command(*args):
    """`nice python -m <args>` in main's venv, to be run with cwd = PANEL_REPO."""
    return ["nice", "-n", str(config.NICE), str(config.MAIN_PYTHON), "-m", *map(str, args)]


def tail(text, lines=8, width=1200):
    kept = [line for line in (text or "").strip().splitlines() if line.strip()][-lines:]
    return "\n".join(kept)[-width:] or "no output"


# --- the CPU governor ---------------------------------------------------------------------------

class CpuGovernor:
    """How many extraction workers may run right now.

    allowed = min(SELFDESC_WORKERS, cores - HEADROOM - other users' load), recomputed every
    GOVERNOR_INTERVAL. Other users' load is the 1-minute load average minus our own running
    workers, so the run neither counts itself as somebody else nor ignores a colleague's job.
    The PAUSE file, a dashboard pause, or a volume below the disk floor makes it 0.
    """

    def __init__(self, state, out_dir):
        self.state = state
        self.out_dir = Path(out_dir)
        self.running = 0
        self._lock = threading.Lock()
        self._allowed = config.SELFDESC_WORKERS
        self._reason = None
        self._checked = 0.0
        self._load = 0.0
        self._disk = {}

    def enter(self):
        with self._lock:
            self.running += 1

    def leave(self):
        with self._lock:
            self.running -= 1

    def refresh(self, force=False):
        now = time.time()
        if not force and now - self._checked < config.GOVERNOR_INTERVAL:
            return
        self._checked = now
        self._load = os.getloadavg()[0]
        with self._lock:
            ours = self.running
        others = max(0.0, self._load - ours)
        allowed = int(min(config.SELFDESC_WORKERS,
                          config.NPROC - config.SELFDESC_HEADROOM - others))
        reason = None
        if allowed < config.SELFDESC_WORKERS:
            # The load figure stays out of the message: it changes every poll, and the log should
            # record changes in what the run does, not every wobble of the load average.
            reason = f"yielding to other load: {max(allowed, 0)} worker(s) allowed"

        self._disk = {}
        for label, path in (("out", self.out_dir), ("data", Path(config.DATA_ROOT))):
            try:
                free = shutil.disk_usage(path).free
            except OSError:
                continue
            self._disk[label] = free
            if free < config.MIN_FREE_BYTES:
                allowed, reason = 0, (f"{label} volume has {free / 1e9:.1f} GB free, below the "
                                      f"{config.MIN_FREE_BYTES / 1e9:.0f} GB floor")
        if config.PAUSE_FILE.exists():
            allowed, reason = 0, f"courtesy PAUSE file at {config.PAUSE_FILE}"

        allowed = max(0, allowed)
        if reason != self._reason:
            self.state.log("warn" if reason else "info",
                           (f"{reason} (load {self._load:.1f}, ours {ours})" if reason
                            else f"full allowance again ({config.SELFDESC_WORKERS} workers)"),
                           "governor")
        self._allowed, self._reason = allowed, reason

    def allowed(self):
        self.refresh()
        return 0 if self.state.is_paused() else self._allowed

    def status(self):
        return {"unit": "cpu", "load": round(self._load, 2),
                "load_pause": config.NPROC - config.SELFDESC_HEADROOM,
                "blocked": {"workers": self._reason} if self._reason else {},
                "running": self.running, "allowed": self._allowed,
                "disk_free_gb": {k: round(v / 1e9, 1) for k, v in self._disk.items()},
                "disk_floor_gb": round(config.MIN_FREE_BYTES / 1e9), "memory": {}}


# --- stages ------------------------------------------------------------------------------------

class Job:
    """Paths and the plan for one run (one manifest x one output directory)."""

    def __init__(self, manifest, out_dir, smoke, channels=("luma",), temporal_configs=None,
                 suite="full"):
        self.manifest = Path(manifest)
        self.out_dir = Path(out_dir)
        self.smoke = smoke
        self.channels = list(channels)
        self.temporal_configs = list(temporal_configs or [config.SELFDESC_FIRST])
        self.suite = suite
        self.plan_path = config.WORK_DIR / "selfdesc-plan.json"
        self.shard = 16 if smoke else config.SELFDESC_SHARD
        self.configs = []      # filled by prepare(): [{config, fre, B, L, dim, ...}]

    def extract_args(self, cfg):
        return ["selfdesc.extract", "--manifest", self.manifest, "--fre", cfg["fre"],
                "--B", cfg["B"], "--L", cfg["L"], "--channels", cfg.get("channels", "luma"),
                "--out", self.out_dir, "--shard-size", self.shard]

    def cfg_dir(self, name):
        return self.out_dir / "descriptions" / name

    def assembled(self, name):
        return (self.cfg_dir(name) / "meta.json").exists()

    def shard_file(self, name, seq):
        return self.cfg_dir(name) / "shards" / f"{seq:05d}.npz"


def run_main(state, args, source, timeout=None):
    """Run one main-side command to completion. Returns (ok, elapsed, output tail)."""
    started = time.perf_counter()
    try:
        result = subprocess.run(main_command(*args), cwd=config.PANEL_REPO, env=child_env(),
                                capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, time.perf_counter() - started, f"timed out after {timeout}s"
    elapsed = time.perf_counter() - started
    out = (result.stdout or "") + (result.stderr or "")
    return result.returncode == 0, elapsed, tail(out)


def prepare(state, job, only=None):
    args = ["selfdesc.sweep", "--manifest", job.manifest, "--out", job.out_dir,
            "--fre-dir", config.SELFDESC_FRE_DIR, "--frozen", "--plan-json", job.plan_path,
            "--channels", *job.channels]
    if only:
        args += ["--only", *only]
    ok, elapsed, message = run_main(state, args, "prepare", timeout=3600)
    if not ok:
        state.log("error", f"prepare failed: {message}", "prepare")
        sys.exit(f"prepare failed:\n{message}")
    job.configs = json.loads(job.plan_path.read_text())
    # K8 first; the rest cheapest-first by the dominant cost, (K*(B^2-1))^2 per row.
    job.configs.sort(key=lambda c: (c["config"] != config.SELFDESC_FIRST, c["dim"] ** 2))
    state.log("info", f"prepared {len(job.configs)} config(s) in {elapsed:.0f}s: "
                      + ", ".join(c["config"] for c in job.configs), "prepare")


def plan(state, job, n_rows):
    for cfg in job.configs:
        rows = [{"detector": cfg["config"], "seq": seq, "start": start,
                 "n": min(job.shard, n_rows - start)}
                for seq, start in enumerate(range(0, n_rows, job.shard))]
        state.add_chunks(rows)
        print(f"  {cfg['config']:<26} {len(rows):>4} shards of up to {job.shard}")


def reconcile(state, job):
    """Chunk status from disk: a shard file, or an assembled config, means done."""
    with state._lock:
        rows = state._db.execute("SELECT id, detector, seq, status FROM chunks").fetchall()
    to_done, to_pending = [], []
    for row in rows:
        present = job.assembled(row["detector"]) or job.shard_file(row["detector"],
                                                                   row["seq"]).exists()
        if row["status"] == state_module.DONE and not present:
            to_pending.append(row["id"])
        elif row["status"] != state_module.DONE and present:
            to_done.append(row["id"])
    with state._lock:
        state._db.executemany("UPDATE chunks SET status = 'pending', updated_at = ? WHERE id = ?",
                              [(time.time(), i) for i in to_pending])
        state._db.executemany("UPDATE chunks SET status = 'done', updated_at = ? WHERE id = ?",
                              [(time.time(), i) for i in to_done])
        state._db.commit()
    print(f"reconcile: {len(to_done)} found on disk marked done, {len(to_pending)} requeued")


def worker(index, state, gov, stop, job, by_name):
    """Claim and compute shards while this worker's index is inside the governor's allowance."""
    while not stop.is_set():
        if index >= gov.allowed():
            # Outside the allowance: wait for it to grow, but leave once there is nothing left to
            # claim, or the run would never get past extraction.
            if not state.pending_exists() or stop.wait(5.0):
                break
            continue
        names = [c["config"] for c in job.configs if not state.is_paused(c["config"])]
        if not names:                      # every config paused from the dashboard
            if stop.wait(5.0):
                break
            continue
        chunk = state.claim(names)
        if chunk is None:
            break
        cfg = by_name[chunk["detector"]]
        gov.enter()
        try:
            ok, elapsed, message = run_main(
                state, [*job.extract_args(cfg), "--shard", chunk["seq"]], cfg["config"],
                timeout=config.SHARD_TIMEOUT)
        finally:
            gov.leave()
        if ok and job.shard_file(cfg["config"], chunk["seq"]).exists():
            state.finish(chunk["id"], state_module.DONE, elapsed_s=elapsed)
        else:
            if ok:
                message = "exited 0 but wrote no shard"
            state.finish(chunk["id"], state_module.FAILED, elapsed_s=elapsed, error=message)
            state.log("error", f"{cfg['config']} shard {chunk['seq']} FAILED: {message}",
                      cfg["config"])
            if not state.exhausted(chunk["id"]):
                state.release(chunk["id"], error=message)


def assemble_ready(state, job):
    """Assemble every config whose shards are all done. Returns the names still outstanding."""
    counts = state.counts()
    outstanding = []
    for cfg in job.configs:
        name = cfg["config"]
        if job.assembled(name):
            continue
        entry = counts.get(name, {})
        if any(entry.get(s, 0) for s in (state_module.PENDING, state_module.ACTIVE,
                                         state_module.FAILED)):
            outstanding.append(name)
            continue
        ok, elapsed, message = run_main(state, [*job.extract_args(cfg), "--assemble"], name,
                                        timeout=3600)
        if ok:
            state.log("info", f"{name}: assembled in {elapsed:.0f}s", name)
        else:
            state.log("error", f"{name}: assemble failed: {message}", name)
            outstanding.append(name)
    return outstanding


def temporal(state, gov, stop, job, perms):
    """Every temporal config in turn. Returns True when all of them finished."""
    for name in job.temporal_configs:
        if stop.is_set() or not temporal_one(state, gov, stop, job, perms, name):
            return False
    return True


def temporal_one(state, gov, stop, job, perms, name):
    """The time tests over one config's descriptions, streamed into the event log."""
    gov.refresh(force=True)
    others = max(0.0, gov._load - gov.running)
    jobs = int(max(1, min(config.SELFDESC_TEMPORAL_JOBS,
                          config.NPROC - config.SELFDESC_HEADROOM - others)))
    args = ["selfdesc.temporal", "--exp", job.out_dir, "--config", name, "--suite", job.suite,
            "--perms", perms, "--jobs", jobs]
    if job.smoke:
        # every block of the real suite, at permutation counts that only prove it runs
        args += ["--grid-perms", 5, "--config-perms", 3, "--power-perms", 5, "--sims", 10,
                 "--boot", 50]
    state.log("info", f"temporal {name} ({job.suite}): starting with {jobs} jobs, {perms} "
                      f"permutations", "temporal")
    process = subprocess.Popen(main_command(*args), cwd=config.PANEL_REPO, env=child_env(),
                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    started = time.time()
    lines = []
    for line in process.stdout:
        line = line.rstrip()
        lines.append(line)
        if line.startswith(("[", "T2 ", "T5 ", "L1 ", "L2 ", "S1 ", "power ", "wrote ")):
            state.log("info", line[:400], "temporal")
        if stop.is_set() or time.time() - started > config.TEMPORAL_TIMEOUT:
            process.terminate()
            state.log("warn", "temporal: stopped; cached blocks resume on the next start",
                      "temporal")
            break
    code = process.wait()
    if code != 0 and not stop.is_set():
        state.log("error", f"temporal failed (exit {code}): {tail(chr(10).join(lines))}",
                  "temporal")
        return False
    return code == 0


# --- the run ----------------------------------------------------------------------------------

def smoke_manifest(manifest, path, per_generator=12, authentic=40):
    """A small manifest that still has every generator: the temporal stage needs >= 5 images per
    generator (and >= 10 per generator x origin cell for T2), which `head(n)` on a RAISE-first
    manifest would never give it."""
    fakes = manifest[manifest["label"] == 1].groupby("generator", sort=False).head(per_generator)
    real = manifest[manifest["label"] == 0].groupby("origin_dataset", sort=False) \
        .head(max(1, authentic // max(1, manifest.loc[manifest.label == 0,
                                                      "origin_dataset"].nunique())))
    subset = pd.concat([real, fakes]).sort_index()
    path.parent.mkdir(parents=True, exist_ok=True)
    subset.to_csv(path, index=False)
    return path


def serve(state, gov, stop, started, job, stage):
    try:
        import uvicorn

        import server as server_module
    except ImportError as error:
        state.log("warn", f"dashboard unavailable ({error}); running without it", "server")
        return
    order = [c["config"] for c in job.configs]
    app = server_module.build_app(
        state, gov, stop, started, order=order, workers=lambda: max(1, gov.running),
        title="Self-descriptions — extraction + temporal tests",
        extra=lambda: {"title": "Self-descriptions — extraction + temporal tests",
                       "stage": stage["now"], "unit_noun": "descriptions",
                       "workers_label": f"{gov.running} running / {gov._allowed} allowed "
                                        f"(max {config.SELFDESC_WORKERS}) cpu workers"})
    instance = uvicorn.Server(uvicorn.Config(app, host=config.HOST, port=config.SELFDESC_PORT,
                                             log_level="warning"))
    threading.Thread(target=instance.run, name="dashboard", daemon=True).start()
    state.log("info", f"dashboard on http://{config.HOST}:{config.SELFDESC_PORT}  (ssh -F ssh_config "
                      f"-L {config.SELFDESC_PORT}:127.0.0.1:{config.SELFDESC_PORT} remote)",
              "server")


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="experiment directory in main, e.g. experiments/selfdesc-36k")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse every stage on a few images per generator, two configs, "
                             "16-row shards and a quick temporal pass")
    parser.add_argument("--configs", nargs="*", default=None, metavar="CONFIG",
                        help="restrict extraction to these config ids")
    parser.add_argument("--channels", nargs="+", default=["luma"], choices=["luma", "rgb"],
                        help="planes to describe (selfdesc.sweep --channels); rgb configs carry "
                             "an _rgb suffix")
    parser.add_argument("--temporal-configs", nargs="+", default=None, metavar="CONFIG",
                        help=f"configs the temporal stage runs on (default "
                             f"{config.SELFDESC_FIRST})")
    parser.add_argument("--suite", choices=["full", "arms", "quick"], default="full",
                        help="selfdesc.temporal --suite")
    parser.add_argument("--stages", nargs="+", default=["extract", "temporal"],
                        choices=["extract", "temporal"])
    parser.add_argument("--perms", type=int, default=config.TEMPORAL_PERMS)
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--assemble-only", action="store_true",
                        help="assemble whatever configs are complete, then exit")
    parser.add_argument("--no-server", action="store_true")
    arguments = parser.parse_args()

    missing = [p for p in (config.MAIN_PYTHON, config.PANEL_REPO / "selfdesc" / "temporal.py",
                           config.SELFDESC_FRE_DIR / "lambda.json") if not Path(p).exists()]
    if missing:
        sys.exit("missing from the main checkout (PANEL_REPO) -- pull main there first:\n  "
                 + "\n  ".join(map(str, missing)))

    manifest_path = arguments.manifest.resolve()
    manifest_sha = sha256_file(manifest_path)
    config.scope_to_manifest(manifest_sha, prefix="selfdesc")
    config.ensure_dirs()
    if arguments.smoke:
        full = pd.read_csv(manifest_path, low_memory=False)
        manifest_path = smoke_manifest(full, config.WORK_DIR / "selfdesc-smoke-manifest.csv")
        configs = arguments.configs or [config.SELFDESC_FIRST, "K1_M11_B11_L3"]
        perms = min(arguments.perms, 20)
        # A smoke manifest is a different file: give it its own state, never the real run's.
        config.scope_to_manifest(sha256_file(manifest_path), prefix="selfdesc-smoke")
        config.ensure_dirs()
    else:
        configs, perms = arguments.configs, arguments.perms

    out_dir = arguments.out_dir.resolve()
    if arguments.smoke and not out_dir.name.endswith("-smoke"):
        # A smoke run writes images.parquet for its subset manifest; sharing the real run's
        # directory would overwrite the real index.
        out_dir = out_dir.with_name(out_dir.name + "-smoke")
    job = Job(manifest_path, out_dir, arguments.smoke, arguments.channels,
              arguments.temporal_configs, arguments.suite)
    n_rows = sum(1 for _ in open(manifest_path)) - 1
    print("\n".join([f"main repo  {config.PANEL_REPO}", f"manifest   {manifest_path}",
                     f"rows       {n_rows}{'  (smoke)' if arguments.smoke else ''}",
                     f"outputs    {job.out_dir}", f"state      {config.STATE_DB}",
                     f"frozen FRE {config.SELFDESC_FRE_DIR}",
                     f"channels   {' '.join(job.channels)}",
                     f"temporal   {' '.join(job.temporal_configs)} (suite {job.suite})",
                     f"workers    up to {config.SELFDESC_WORKERS} (headroom "
                     f"{config.SELFDESC_HEADROOM}, nice {config.NICE})",
                     f"dashboard  http://{config.HOST}:{config.SELFDESC_PORT}", ""]))

    state = state_module.State()
    print("--- prepare ---")
    prepare(state, job, configs)
    by_name = {c["config"]: c for c in job.configs}

    if arguments.assemble_only:
        print("outstanding:", assemble_ready(state, job) or "none")
        return

    stop = threading.Event()

    def handle_signal(signum, _frame):
        print(f"\nsignal {signum} -- finishing the shards in flight, then stopping", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    gov = CpuGovernor(state, job.out_dir)
    gov.refresh(force=True)
    stage = {"now": "prepare"}
    started = time.time()
    if not arguments.no_server:
        serve(state, gov, stop, started, job, stage)

    if "extract" in arguments.stages:
        print("--- planning ---")
        plan(state, job, n_rows)
        if state.requeue_active():
            print("requeued shards left active by a previous run")
        if arguments.retry_failed:
            print(f"requeued {state.retry_failed()} failed shards")
        reconcile(state, job)

        stage["now"] = "extract"
        print(f"\n--- extracting with up to {config.SELFDESC_WORKERS} cpu workers ---\n")
        threads = [threading.Thread(target=worker, args=(i, state, gov, stop, job, by_name),
                                    name=f"cpu{i}", daemon=True)
                   for i in range(config.SELFDESC_WORKERS)]
        for thread in threads:
            thread.start()
        # Assemble configs as they complete rather than at the end, so K8's descriptions exist
        # (and can be looked at) hours before the slowest config finishes.
        last_assembly = time.time()
        while any(t.is_alive() for t in threads):
            time.sleep(2.0)
            if time.time() - last_assembly > 60:
                assemble_ready(state, job)
                last_assembly = time.time()
        stage["now"] = "assemble"
        outstanding = assemble_ready(state, job)
        for name, entry in sorted(state.counts().items()):
            per = state.per_image_seconds(name)
            print(f"  {name:<26} {entry.get('images_done', 0):>7}/{entry.get('images_total', 0):<7}"
                  f" {f'{per * 1000:.0f} ms/img/worker' if per else '-'}")
        if outstanding:
            failures = state.failures()
            print(f"\nnot assembled: {outstanding}")
            for failure in failures:
                print(f"  {failure['detector']} shard {failure['seq']} "
                      f"(attempt {failure['attempts']}): {failure['error']}")
            if stop.is_set():
                print("stopped; restart with the same command to resume")
            sys.exit(1)

    if "temporal" in arguments.stages and not stop.is_set():
        stage["now"] = "temporal"
        unassembled = [c for c in job.temporal_configs if not job.assembled(c)]
        if unassembled:
            sys.exit(f"not assembled, the temporal stage needs them: {unassembled}")
        ok = temporal(state, gov, stop, job, perms)
        stage["now"] = "done" if ok else "temporal failed"
        print(f"\n--- temporal {'finished' if ok else 'did not finish'} after "
              f"{(time.time() - started) / 60:.1f} min ---")
        if not ok:
            sys.exit(1)
    state.close()


if __name__ == "__main__":
    main()
