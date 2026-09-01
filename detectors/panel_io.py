"""Shared plumbing for the detector panel's run_score.py scripts.

Pure stdlib + pandas, so it imports cleanly in all four isolated environments without
adding a dependency to any of them. Each run_score.py adds this file's directory to
sys.path and imports it; nothing here knows anything about a specific detector.

The contract every run_score.py honours:
  * reads `crop_path` from the manifest (falling back to `path`),
  * processes rows in manifest order,
  * writes exactly `image_id,raw_score`,
  * emits a <out>.meta.json sidecar recording what produced the scores,
  * orients raw_score so that HIGHER MEANS MORE SYNTHETIC.
"""

import argparse
import hashlib
import json
import platform
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def add_common_args(parser):
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--device", default="cpu", help="cpu | mps (default cpu: bit-reproducible)")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--limit", type=int, default=None, help="score only the first N rows (smoke tests)")
    return parser


def resolve(rel):
    path = Path(rel)
    return path if path.is_absolute() else ROOT / path


def load_manifest(manifest_path, limit=None):
    """Manifest rows plus an `image_file` column holding the absolute path to score.

    Prefers the preprocessed crop. A missing crop is fatal rather than skippable: a short
    scores file would still merge, and would then quietly corrupt downstream calibration.
    """
    manifest = pd.read_csv(manifest_path)
    if limit is not None:
        manifest = manifest.head(limit)

    column = "crop_path" if "crop_path" in manifest.columns else "path"
    manifest = manifest.copy()
    manifest["image_file"] = [resolve(p) for p in manifest[column]]

    if not manifest["image_id"].is_unique:
        dupes = manifest.loc[manifest["image_id"].duplicated(), "image_id"].tolist()
        raise SystemExit(f"duplicate image_id in {manifest_path}: {dupes[:5]}")

    missing = [str(p) for p in manifest["image_file"] if not p.exists()]
    if missing:
        raise SystemExit(
            f"{len(missing)} image(s) from {manifest_path} are missing, first: {missing[0]}\n"
            f"rebuild the crop cache with:\n"
            f"  .venv/bin/python imports/preprocess_crop_cache.py --manifest {manifest_path}"
        )
    return manifest


def rel_to_root(path):
    """Repo-relative when possible, absolute otherwise (smoke-test manifests live outside)."""
    path = Path(path).resolve()
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit(repo_dir):
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_dir), "rev-parse", "HEAD"], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:
        return None


def write_scores(out_path, image_ids, scores):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"image_id": image_ids, "raw_score": scores}).to_csv(out_path, index=False)


def write_meta(out_path, detector, upstream_repo, upstream_dir, weights, score_semantics,
               manifest_path, device, n_rows, elapsed_s, extra=None):
    """Sidecar recording the triple that makes a score file reproducible.

    manifest sha256 + crop policy + panel version. Without it, comparing experiment 02's
    scores against experiment 01's is guesswork, and the attestation emitter has no panel
    manifest to chain to.
    """
    import torch

    manifest = pd.read_csv(manifest_path)
    crop_policy = None
    if "crop_path" in manifest.columns and len(manifest):
        crop_policy = Path(manifest["crop_path"].iloc[0]).parent.name

    meta = {
        "detector": detector,
        "score_semantics": score_semantics,
        "score_orientation": "higher_is_more_synthetic",
        "upstream_repo": upstream_repo,
        "upstream_commit": git_commit(upstream_dir),
        "weights": [{"path": rel_to_root(w), "sha256": sha256_file(w)} for w in weights],
        "manifest": {
            "path": rel_to_root(manifest_path),
            "sha256": sha256_file(manifest_path),
            "rows": n_rows,
        },
        "crop_policy": crop_policy,
        "runtime": {
            "device": device,
            "torch": torch.__version__,
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "elapsed_s": round(elapsed_s, 1),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    if extra:
        meta.update(extra)

    meta_path = out_path.with_suffix(".meta.json")
    meta_path.write_text(json.dumps(meta, indent=2) + "\n")
    return meta_path


class Timer:
    def __enter__(self):
        self.start = time.perf_counter()
        return self

    def __exit__(self, *exc):
        self.elapsed = time.perf_counter() - self.start


def report(out_path, scores, elapsed):
    import numpy as np

    arr = np.asarray(scores, dtype=float)
    print(
        f"wrote {len(arr)} scores -> {out_path}  "
        f"[{elapsed:.1f}s, min={arr.min():.4f} median={np.median(arr):.4f} max={arr.max():.4f}]"
    )
