"""Score a manifest with the whole panel, unattended and resumably. Run this; go to sleep.

    python -m runner.pipeline --manifest m.csv --out-dir experiments/x --smoke --limit 40
    python -m runner.pipeline --manifest m.csv --out-dir experiments/x

One worker per GPU, pulling chunks from a shared queue ordered cheapest-detector-first, so a run
that dies overnight leaves whole score files rather than five partial ones.

Interrupting is safe at any point. Chunk outputs are staged and moved into place, so a file that
exists is complete; anything left `active` goes back to `pending` on the next start; and the
reconcile pass makes the output directory authoritative, so even losing the state database costs
a rescan rather than a re-score.
"""

import argparse
import hashlib
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd  # noqa: E402

import config  # noqa: E402
import governor as governor_module  # noqa: E402
import merge  # noqa: E402
import state as state_module  # noqa: E402
import worker  # noqa: E402


def sha256_file(path, block=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for piece in iter(lambda: handle.read(block), b""):
            digest.update(piece)
    return digest.hexdigest()


def plan(state, manifest, members):
    """Populate the chunk queue and write the chunk manifests. Idempotent.

    Chunk manifests are keyed by size rather than by detector, so the four members that share a
    chunk size share the files instead of writing four identical copies of every slice.
    """
    total = 0
    for member in members:
        rows = []
        for seq, start in enumerate(range(0, len(manifest), member.chunk)):
            piece = manifest.iloc[start:start + member.chunk]
            path = state_module.chunk_manifest_path(member.chunk, seq)
            if not path.exists():
                path.parent.mkdir(parents=True, exist_ok=True)
                staging = path.with_suffix(".csv.part")
                piece.to_csv(staging, index=False)
                staging.replace(path)
            rows.append({"detector": member.name, "seq": seq, "start": start, "n": len(piece)})
        state.add_chunks(rows)
        total += len(rows)
        print(f"  {member.name:<26} {len(rows):>3} chunks of up to {member.chunk}")
    return total


def serve(state, gov, stop, started):
    """Dashboard in a background thread. Its failure must never stop the run."""
    try:
        import uvicorn

        import server as server_module
    except ImportError as error:
        state.log("warn", f"dashboard unavailable ({error}); scoring without it", "server")
        return None

    app = server_module.build_app(state, gov, stop, started)
    settings = uvicorn.Config(app, host=config.HOST, port=config.PORT, log_level="warning")
    instance = uvicorn.Server(settings)
    threading.Thread(target=instance.run, name="dashboard", daemon=True).start()
    state.log("info", f"dashboard on http://{config.HOST}:{config.PORT}"
                      f"  (ssh -F ssh_config -L {config.PORT}:127.0.0.1:{config.PORT} remote)",
              "server")
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True,
                        help="where the merged <detector>.csv files land")
    parser.add_argument("--detectors", nargs="+", choices=config.PANEL_ORDER,
                        default=config.PANEL_ORDER)
    parser.add_argument("--gpus", type=int, nargs="+", default=config.GPUS)
    parser.add_argument("--device", default="cuda",
                        help="passed through to run_score.py. cpu is for rehearsing the "
                             "orchestration on a machine without a GPU, not for a real run")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse the whole path on --limit images with tiny chunks")
    parser.add_argument("--limit", type=int, default=40, help="images in --smoke mode")
    parser.add_argument("--plan-only", action="store_true")
    parser.add_argument("--reconcile", action="store_true",
                        help="rebuild chunk status from the output directory, then exit")
    parser.add_argument("--retry-failed", action="store_true")
    parser.add_argument("--merge-only", action="store_true",
                        help="assemble whatever chunks are already done, then exit")
    parser.add_argument("--no-server", action="store_true")
    arguments = parser.parse_args()

    members = [config.PANEL_BY_NAME[name] for name in arguments.detectors]

    # fp16 is a CUDA-only path in aeroblade's run_score.py, and it refuses rather than silently
    # falling back. Rehearsing on CPU should exercise the orchestration, not die on that guard.
    if not arguments.device.startswith("cuda"):
        for member in members:
            if "fp16" in member.extra:
                member.extra = ["fp32" if a == "fp16" else a for a in member.extra]
                print(f"note: {member.name} --dtype fp16 -> fp32 for --device {arguments.device}")

    manifest = pd.read_csv(arguments.manifest)
    if arguments.smoke:
        manifest = manifest.head(arguments.limit)
        for member in members:
            member.chunk = max(1, arguments.limit // 2)   # at least two chunks, to exercise resume
    manifest_sha = sha256_file(arguments.manifest)

    # Before ensure_dirs and before the database is opened: both depend on the scoped paths.
    config.scope_to_manifest(manifest_sha)
    config.ensure_dirs()

    print(config.describe(arguments.manifest, arguments.out_dir))
    print(f"rows       {len(manifest)}{'  (smoke)' if arguments.smoke else ''}\n")

    db = state_module.State()

    if arguments.reconcile:
        state_module.reconcile(db)
        return
    if arguments.merge_only:
        for name, status in merge.merge_all(db, manifest, arguments.out_dir,
                                            arguments.manifest, manifest_sha,
                                            arguments.detectors).items():
            print(f"  {name:<26} {status}")
        return

    requeued = db.requeue_active()
    if requeued:
        print(f"requeued {requeued} chunks left active by a previous run")
    if arguments.retry_failed:
        print(f"requeued {db.retry_failed()} previously failed chunks")

    print("--- planning ---")
    plan(db, manifest, members)
    state_module.reconcile(db)
    if arguments.plan_only:
        print(db.snapshot_json())
        return

    stop = threading.Event()
    gov = governor_module.Governor(db)
    gov.refresh(force=True)

    def handle_signal(signum, _frame):
        print(f"\nsignal {signum} -- finishing the chunks in flight, then stopping", flush=True)
        stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    started = time.time()
    if not arguments.no_server:
        serve(db, gov, stop, started)

    print(f"\n--- scoring on gpu {arguments.gpus} ---\n")
    threads = []
    for gpu in arguments.gpus:
        thread = threading.Thread(
            target=worker.run_worker,
            args=(gpu, db, gov, stop, arguments.out_dir, arguments.detectors, arguments.device),
            name=f"gpu{gpu}", daemon=True)
        thread.start()
        threads.append(thread)
        time.sleep(1.0)   # stagger the two model loads off each other

    for thread in threads:
        while thread.is_alive():
            thread.join(timeout=1.0)

    print(f"\n--- stopped after {(time.time() - started) / 60:.1f} min ---")
    for name, entry in sorted(db.counts().items()):
        per_image = db.per_image_seconds(name)
        rate = f"{per_image * 1000:.0f} ms/img" if per_image else "-"
        print(f"  {name:<26} {entry.get('images_done', 0):>7}/{entry.get('images_total', 0):<7} {rate}")

    failures = db.failures()
    if failures:
        print(f"\n{len(failures)} failed chunk(s):")
        for failure in failures:
            print(f"  {failure['detector']} chunk {failure['seq']} "
                  f"(attempt {failure['attempts']}): {failure['error']}")

    if not stop.is_set():
        print("\n--- merging ---")
        for name, status in merge.merge_all(db, manifest, arguments.out_dir,
                                            arguments.manifest, manifest_sha,
                                            arguments.detectors).items():
            print(f"  {name:<26} {status}")

    db.close()


if __name__ == "__main__":
    main()
