"""Assemble per-chunk score files into the two artifacts the rest of the project expects.

For each detector: `<out_dir>/<detector>.csv` with exactly `image_id,raw_score` in manifest order,
and a `.meta.json` sidecar. Chunking is an implementation detail of how the run was executed; it
must not leak into the provenance record, so the sidecar is reconstructed to look like one run --
with the fields that genuinely varied per chunk (elapsed, per-chunk manifest hash) collapsed or
replaced by the parent manifest's.

The invariants checked here are the ones whose violation would be silent: that every chunk agrees
on the weights and upstream commit it used, and that the concatenated image_ids are exactly the
manifest's, in order. A score file that is merely *shorter* than its manifest would still join
downstream, and would corrupt calibration without ever raising.
"""

import json
from datetime import datetime, timezone

import pandas as pd

import config
import state as state_module

# Per-chunk by nature; collapsed rather than compared.
VARIES_PER_CHUNK = {"manifest", "elapsed_s", "generated_at"}


def merge_detector(detector, manifest, out_dir, manifest_path, manifest_sha, wall_seconds=None,
                   chunk_size=None):
    chunks = sorted((config.WORK_DIR / "scores" / detector).glob("[0-9]*.csv"))
    if not chunks:
        return None, "no chunks"

    frames = [pd.read_csv(path) for path in chunks]
    scores = pd.concat(frames, ignore_index=True)

    if len(scores) != len(manifest):
        return None, f"{len(scores)} scores against {len(manifest)} manifest rows"
    mismatched = (scores["image_id"].values != manifest["image_id"].values).sum()
    if mismatched:
        return None, f"{mismatched} image_ids do not match the manifest, in order"

    out_dir.mkdir(parents=True, exist_ok=True)
    target = out_dir / f"{detector}.csv"
    scores.to_csv(target, index=False)

    sidecars = [path.with_suffix(".meta.json") for path in chunks]
    meta, problem = merge_meta(detector, sidecars, manifest_path, manifest_sha, len(scores),
                               wall_seconds, chunk_size)
    if problem:
        return target, problem
    (out_dir / f"{detector}.meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    return target, None


def merge_meta(detector, sidecars, manifest_path, manifest_sha, n_rows, wall_seconds=None,
               chunk_size=None):
    """One run-level sidecar from many per-chunk ones, refusing to paper over disagreement."""
    present = [path for path in sidecars if path.exists()]
    if not present:
        return None, "no chunk sidecars to merge"

    metas = [json.loads(path.read_text()) for path in present]
    base = dict(metas[0])

    # Anything that is not per-chunk must be identical across chunks. A differing weight hash or
    # upstream commit means the panel changed underneath the run, which invalidates the whole
    # score file rather than the one chunk it was noticed in.
    for key in set(base) - VARIES_PER_CHUNK:
        values = {json.dumps(m.get(key), sort_keys=True) for m in metas}
        if len(values) > 1:
            return None, f"chunks disagree on {key!r}: {sorted(values)[:2]}"

    # Two different clocks, and conflating them overstates throughput several-fold. Each detector's
    # own Timer covers its scoring loop only -- for cnndetection that excludes imports, CUDA init
    # and the model load, which is most of a short chunk. `wall_seconds` is what the runner
    # measured around the whole subprocess, and is the number to plan a future run from.
    scoring = sum(m.get("elapsed_s") or 0 for m in metas)
    base["manifest"] = {"path": str(manifest_path), "sha256": manifest_sha, "rows": n_rows}
    base["elapsed_s"] = round(scoring, 1)
    base["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    base["execution"] = {
        "mode": "chunked",
        "chunks": len(metas),
        # Needed to reproduce the scores, not just to describe the run. Measured 2026-09-26:
        # re-running a manifest at the same chunk and batch size is bit-identical, on either
        # GPU. Changing the chunk size is not -- it moves images into differently shaped
        # batches, cuDNN selects kernels per shape, and ~23% of scores shift by up to 0.1% of
        # the score range. Irrelevant to any conclusion, fatal to a byte-for-byte comparison,
        # so the geometry that produced these numbers is recorded alongside them.
        "chunk_size": chunk_size,
        "scoring_seconds": round(scoring, 1),
        "wall_seconds": round(wall_seconds, 1) if wall_seconds else None,
        "images_per_second": round(n_rows / wall_seconds, 2) if wall_seconds else None,
        "note": "scored in chunks by runner/. scoring_seconds is the sum of each chunk's own "
                "timer (inference only); wall_seconds is the sum of subprocess wall times and "
                "includes the per-chunk model load. Both are sums across workers, so with two "
                "GPUs they exceed the run's wall clock.",
    }
    return base, None


def merge_all(state, manifest, out_dir, manifest_path, manifest_sha, detectors=None):
    """Merge every detector whose chunks are all done. Returns {detector: status string}."""
    results = {}
    counts = state.counts()
    for name in detectors or config.PANEL_ORDER:
        entry = counts.get(name)
        if not entry:
            continue
        if entry.get(state_module.DONE, 0) == 0:
            results[name] = "nothing done yet"
            continue
        outstanding = sum(entry.get(s, 0) for s in (state_module.PENDING,
                                                    state_module.ACTIVE,
                                                    state_module.FAILED))
        if outstanding:
            results[name] = f"incomplete ({outstanding} chunks outstanding) -- not merged"
            continue

        wall = sum(c["elapsed_s"] or 0 for c in state.chunks_for(name, state_module.DONE))
        target, problem = merge_detector(name, manifest, out_dir, manifest_path, manifest_sha,
                                         wall, config.PANEL_BY_NAME[name].chunk)
        if problem and target is None:
            results[name] = f"FAILED: {problem}"
        elif problem:
            results[name] = f"scores written, sidecar skipped: {problem}"
        else:
            results[name] = f"-> {target}"
    return results
