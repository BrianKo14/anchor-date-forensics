"""One GPU worker: claim a chunk, score it in the detector's own venv, record the result.

Every chunk is a fresh `run_score.py` subprocess. That is deliberate, and it is what keeps the
frozen panel frozen: the detector scripts are invoked exactly as run_all.sh invokes them, with the
same arguments and the same per-run sidecar, so a score produced here is the same score the Mac
would produce. Nothing in this package imports torch or touches detector code.

The cost is reloading the model once per chunk -- seconds against the minutes a chunk takes, and
the price of not forking five upstream-adjacent scripts into a worker-loop variant that would then
have to be kept in step with them.
"""

import os
import shutil
import subprocess
import time

import config
import state as state_module


def chunk_env(gpu, scratch):
    """Environment for one chunk's subprocess.

    CUDA_VISIBLE_DEVICES rather than --device cuda:N so the child sees exactly one card and any
    bare .cuda() or "cuda" device string inside upstream code lands on the card we intended.
    Renumbering means the child always calls it cuda:0.

    The three cache variables all point at /data for one reason: every one of them defaults to
    somewhere on the 492 GB root volume this machine shares with eighteen other home directories.
    HF_HUB_CACHE and TORCH_HOME are ~1.5 GB of model weights. AEROBLADE_CACHE_ROOT is the one that
    actually bit -- see score_chunk.
    """
    env = dict(os.environ)
    env["CUDA_VISIBLE_DEVICES"] = str(gpu)
    env["AEROBLADE_CACHE_ROOT"] = str(scratch)
    env.setdefault("HF_HUB_CACHE", str(config.WORK_DIR.parent / "hf-cache"))
    env.setdefault("TORCH_HOME", str(config.WORK_DIR.parent / "torch-cache"))
    # Each worker already owns a whole GPU; letting torch also spawn 48 CPU threads for the
    # dataloader side just makes the two workers fight over the box.
    env.setdefault("OMP_NUM_THREADS", "8")
    return env


def score_chunk(member, chunk, gpu, out_dir, device="cuda", timeout=None):
    """Run one chunk. Returns (ok, elapsed_s, message).

    Writes into a temp directory and moves the results into place only on success, so a chunk file
    that exists is always a complete chunk -- the invariant reconcile() depends on.
    """
    seq = chunk["seq"]
    manifest = state_module.chunk_manifest_path(member.chunk, seq)
    final = state_module.chunk_score_path(member.name, seq)
    final.parent.mkdir(parents=True, exist_ok=True)

    staging = final.parent / f".tmp-{seq:05d}"
    shutil.rmtree(staging, ignore_errors=True)
    staging.mkdir(parents=True)
    out = staging / final.name

    # Scratch for whatever the detector writes relative to its working directory -- in practice
    # aeroblade's joblib cache, which is write-only here (every chunk scores different images, so
    # the memo never hits) and cost 106 GB across 280,000 images before it filled the shared root
    # filesystem. Kept out of `staging`, which gets moved into place on success; this is deleted
    # either way.
    scratch = config.WORK_DIR / "scratch" / member.name / f"{seq:05d}"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True)

    command = [
        str(member.python), str(member.script),
        "--manifest", str(manifest),
        "--out", str(out),
        "--device", device,
        "--batch-size", str(member.batch),
        *member.extra,
    ]

    started = time.perf_counter()
    try:
        try:
            result = subprocess.run(
                command, env=chunk_env(gpu, scratch), capture_output=True, text=True,
                timeout=timeout or config.SUBPROCESS_TIMEOUT,
            )
        except subprocess.TimeoutExpired:
            return (False, time.perf_counter() - started,
                    f"timed out after {timeout or config.SUBPROCESS_TIMEOUT}s")
        elapsed = time.perf_counter() - started

        if result.returncode != 0:
            return False, elapsed, tail(result.stderr or result.stdout)
        if not out.exists():
            return False, elapsed, "run_score.py exited 0 but wrote no output"

        # A short score file still merges, and would then quietly corrupt calibration -- the same
        # failure panel_io.load_manifest refuses to allow on the way in.
        rows = sum(1 for _ in out.open()) - 1
        if rows != chunk["n"]:
            return False, elapsed, f"expected {chunk['n']} scores, got {rows}"

        for produced in staging.iterdir():
            produced.replace(final.parent / produced.name)
        return (True, elapsed,
                f"{rows} scores in {elapsed:.0f}s ({elapsed / max(rows, 1) * 1000:.0f} ms/img)")
    finally:
        # Both directories go on every path, including timeout and crash. The scratch one in
        # particular: leaving it behind on failure is how a retry loop fills a disk.
        shutil.rmtree(staging, ignore_errors=True)
        shutil.rmtree(scratch, ignore_errors=True)


def tail(text, lines=6, width=800):
    if not text:
        return "no output"
    kept = [line for line in text.strip().splitlines() if line.strip()][-lines:]
    return "\n".join(kept)[-width:]


def run_worker(gpu, state, governor, stop, out_dir, detectors=None, device="cuda"):
    """Claim and score chunks until the queue empties or the run is told to stop."""
    scored = 0
    while not stop.is_set():
        reason = governor.blocked(gpu) if device.startswith("cuda") else None
        if reason or state.is_paused() or config.PAUSE_FILE.exists():
            if stop.wait(5.0):
                break
            continue

        chunk = state.claim(detectors)
        if chunk is None:
            break

        member = config.PANEL_BY_NAME[chunk["detector"]]
        if state.is_paused(member.name):
            state.release(chunk["id"])
            if stop.wait(5.0):
                break
            continue

        state.log("info", f"gpu{gpu} {member.name} chunk {chunk['seq']} "
                          f"({chunk['n']} images) starting", member.name)
        ok, elapsed, message = score_chunk(member, chunk, gpu, out_dir, device)

        if ok:
            state.finish(chunk["id"], state_module.DONE, elapsed_s=elapsed, gpu=gpu)
            state.log("info", f"gpu{gpu} {member.name} chunk {chunk['seq']}: {message}", member.name)
            scored += 1
        else:
            state.finish(chunk["id"], state_module.FAILED, elapsed_s=elapsed, gpu=gpu, error=message)
            state.log("error", f"gpu{gpu} {member.name} chunk {chunk['seq']} FAILED: {message}",
                      member.name)
            if not state.exhausted(chunk["id"]):
                state.release(chunk["id"], error=message)
    return scored
