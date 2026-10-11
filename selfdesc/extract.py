"""Compute forensic self-descriptions for every row of a manifest and write them to disk.

    python -m selfdesc.extract --manifest experiments/selfdesc-sample/manifest.csv \\
        --fre experiments/selfdesc-sample/fre/fre_K8_M11.npz --B 11 --L 3 \\
        --out experiments/selfdesc-sample --workers 8

Usually driven by selfdesc/sweep.py, which also trains the FREs. The expensive part of the
experiment lives here so the notebook only ever loads; the same command runs the in-repo sample
locally and an experiments/<name>/manifest.csv on the server.

Output, under <out>/:

    images.parquet                       one metadata row per manifest row, shared by all configs
    descriptions/<config>/descriptions.npz   X (n, K*(B*B-1)) float32, image_id (n,)
    descriptions/<config>/diagnostics.parquet  per-image fit diagnostics (core.describe)
    descriptions/<config>/meta.json      config, FRE + manifest hashes, crop policy, commit, timing

Resumable: work is written as atomic shards under descriptions/<config>/shards/ and a killed run
picks up where it stopped. run.json pins the config, FRE hash, manifest hash and shard size; a
rerun that disagrees with it refuses to mix shards. Shards are deleted once assembled.
"""

import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from . import core, fre

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "imports" / "sample"))

import common  # noqa: E402

__all__ = ["config_id", "image_index", "extract", "file_sha256"]

# Every image in the sample went through imaging.prepare_image: decoded, alpha-composited onto
# white, re-encoded as JPEG q95. That is the file the crop was cut from.
STORED_FORMAT = f"JPEG q{common.JPEG_QUALITY} (imaging.prepare_image)"

# The authentic importers do not record a source container. Filled only where the source fixes
# it: COCO ships JPEG; the RAISE importer downloads the TIFF column of RAISE_urls.csv. LAION is
# whatever each scraped URL served, so it stays unknown rather than guessed.
AUTHENTIC_SOURCE_FORMAT = {"COCO2017": "JPEG", "COCO2017_train": "JPEG", "COCO2017_val": "JPEG",
                           "RAISE": "TIFF"}


def config_id(fre_path, B, L):
    """fre_K8_M11.npz + B=11, L=3 -> "K8_M11_B11_L3"."""
    return f"{Path(fre_path).stem.removeprefix('fre_')}_B{B}_L{L}"


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def git_commit():
    try:
        return subprocess.run(["git", "-C", str(PROJECT_ROOT), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        return None


def _display_path(path):
    """Repo-relative when inside the repo, so sidecars read the same on every machine."""
    path = Path(path).resolve()
    return str(path.relative_to(PROJECT_ROOT)) if path.is_relative_to(PROJECT_ROOT) else str(path)


def crop_policy(manifest):
    policies = manifest["crop_path"].map(lambda p: Path(p).parent.name).unique()
    assert len(policies) == 1, f"manifest mixes crop policies: {sorted(policies)}"
    return policies[0]


def _sample_metadata():
    """source_format per file_id, from whichever importer manifests exist on this machine.

    The in-repo sample has only the train partition; the server's full build has both.
    """
    frames = []
    for split in ("train", "validation"):
        for out_dir, stem in ((common.AUTHENTIC_DIR, "authentic"), (common.FAKES_DIR, "fakes")):
            path = out_dir / f"{stem}_{split}.parquet"
            if path.exists():
                frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame(columns=["file_id", "source_format"])
    sample = pd.concat(frames, ignore_index=True)
    if "source_format" not in sample.columns:
        sample["source_format"] = None
    return sample[["file_id", "source_format"]].drop_duplicates("file_id")


def image_index(manifest):
    """The metadata every later analysis needs, one row per manifest row, in manifest order.

    generator / family / release date / origin dataset / split come straight from the manifest
    (build_manifest.py is their single source of truth). Size is the image's native size before
    cropping. Format is two things: `source_format`, the container the image arrived in (from
    the importers), and `stored_format`, what every crop was actually cut from.
    """
    index = manifest[["image_id", "label", "generator", "generator_family", "release_date",
                      "origin_dataset", "split", "source_width", "source_height",
                      "crop_top", "crop_left", "crop_height", "crop_width", "crop_path"]].copy()
    meta = _sample_metadata().rename(columns={"file_id": "image_id"})
    index = index.merge(meta, on="image_id", how="left", validate="one_to_one")
    assert len(index) == len(manifest)

    recorded = index["source_format"].notna()
    by_origin = index["origin_dataset"].map(AUTHENTIC_SOURCE_FORMAT)
    fill = ~recorded & (index["label"] == common.REAL_LABEL) & by_origin.notna()
    index.loc[fill, "source_format"] = by_origin[fill]
    index["source_format_basis"] = np.select([recorded, fill], ["importer", "origin_dataset"],
                                             "unknown")
    index["source_format"] = index["source_format"].fillna("unknown")
    index["stored_format"] = STORED_FORMAT
    index["crop_policy"] = crop_policy(manifest)
    index["release_date"] = pd.to_datetime(index["release_date"], errors="coerce")
    return index


# --- workers -----------------------------------------------------------------------------------

def _describe_shard(args):
    """Describe one shard's rows and write the shard atomically. Runs in a worker process."""
    shard_path, rows, fre_path, B, L = args
    filters, _ = fre.load(fre_path)
    K = filters.shape[0]
    X = np.empty((len(rows), core.description_dim(K, B)), dtype=np.float32)
    energy = np.empty((len(rows), K))
    fit_mse = np.empty(len(rows))
    phi_norm = np.empty(len(rows))
    for i, (_, crop_path) in enumerate(rows):
        phi, diag = core.describe(core.load_gray(PROJECT_ROOT / crop_path), filters, B, L)
        X[i] = phi
        energy[i] = diag["residual_energy"]
        fit_mse[i] = diag["fit_mse"]
        phi_norm[i] = diag["phi_norm"]
    tmp = shard_path.with_name(f".{shard_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, X=X, image_id=np.array([image_id for image_id, _ in rows]),
             residual_energy=energy, fit_mse=fit_mse, phi_norm=phi_norm)
    os.replace(tmp, shard_path)
    return len(rows)


def extract(manifest_path, fre_path, B, L, out_dir, workers=1, limit=None, shard_size=None):
    """Describe every manifest row under one config. Returns the config's output directory."""
    manifest_path, fre_path, out_dir = Path(manifest_path), Path(fre_path), Path(out_dir)
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if limit:
        manifest = manifest.iloc[:limit]
    filters, fre_sha = fre.load(fre_path)
    K, M, _ = filters.shape
    cid = config_id(fre_path, B, L)
    cfg_dir = out_dir / "descriptions" / cid
    shard_dir = cfg_dir / "shards"

    # meta.json is written last, so its presence means every other output is complete.
    if (cfg_dir / "meta.json").exists():
        print(f"{cid}: already assembled, skipping")
        return cfg_dir

    n = len(manifest)
    if shard_size is None:
        # Enough shards to keep every worker busy, few enough to stay cheap at 355k rows.
        shard_size = int(min(1024, max(16, -(-n // (4 * max(workers, 1))))))
    run = {"config": cid, "K": K, "M": M, "B": B, "L": L, "fre_sha256": fre_sha,
           "manifest_sha256": file_sha256(manifest_path), "limit": limit, "rows": n,
           "shard_size": shard_size, "crop_policy": crop_policy(manifest)}
    run_path = cfg_dir / "run.json"
    if run_path.exists():
        previous = json.loads(run_path.read_text())
        if previous != run:
            diff = {k: (previous.get(k), run[k]) for k in run if previous.get(k) != run[k]}
            raise SystemExit(f"{cid}: existing shards were made under a different run {diff}; "
                             f"delete {cfg_dir} to start over")
    shard_dir.mkdir(parents=True, exist_ok=True)
    run_path.write_text(json.dumps(run, indent=2) + "\n")

    rows = list(zip(manifest["image_id"], manifest["crop_path"]))
    todo = []
    for s, start in enumerate(range(0, n, shard_size)):
        path = shard_dir / f"{s:05d}.npz"
        if not path.exists():
            todo.append((path, rows[start:start + shard_size], fre_path, B, L))

    started = time.time()
    done = n - sum(len(t[1]) for t in todo)
    print(f"{cid}: {n} rows, D={core.description_dim(K, B)}, {len(todo)} shard(s) to do, "
          f"{done} rows already done")
    if workers > 1 and len(todo) > 1:
        # One BLAS thread per process: the workers already fill the cores, and nested
        # threading on top of them only adds contention. Spawned children read these at import.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(_describe_shard, todo):
                done += count
                print(f"  {done}/{n}  ({time.time() - started:.0f}s)", flush=True)
    else:
        for task in todo:
            done += _describe_shard(task)
            print(f"  {done}/{n}  ({time.time() - started:.0f}s)", flush=True)
    elapsed = time.time() - started

    shards = [np.load(shard_dir / f"{s:05d}.npz") for s in range(-(-n // shard_size))]
    X = np.concatenate([s["X"] for s in shards])
    ids = np.concatenate([s["image_id"] for s in shards])
    assert (ids == manifest["image_id"].to_numpy()).all(), "shard order drifted from the manifest"
    assert np.isfinite(X).all(), f"{cid}: non-finite descriptions"

    diag = pd.DataFrame({"image_id": ids,
                         "fit_mse": np.concatenate([s["fit_mse"] for s in shards]),
                         "phi_norm": np.concatenate([s["phi_norm"] for s in shards])})
    energy = np.concatenate([s["residual_energy"] for s in shards])
    for k in range(K):
        diag[f"residual_energy_{k}"] = energy[:, k]

    np.savez(cfg_dir / "descriptions.npz", X=X, image_id=ids)
    diag.to_parquet(cfg_dir / "diagnostics.parquet", index=False)
    fre_meta_path = fre_path.with_suffix(".meta.json")
    meta = dict(run,
                dim=int(X.shape[1]),
                fre_file=_display_path(fre_path),
                fre_meta=json.loads(fre_meta_path.read_text()) if fre_meta_path.exists() else None,
                manifest=_display_path(manifest_path),
                ridge=core.RIDGE,
                upstream_reference="github.com/ductai199x/Forensic-Self-Descriptions-CVPR25@50f2eae",
                deviations=["no resize to 1024: input is the native-pixel crop (crop_policy)",
                            "lambda calibrated to the released FRE's spectral diversity (fre.py)"],
                git_commit=git_commit(),
                workers=workers,
                seconds_this_run=round(elapsed, 1),
                numpy=np.__version__,
                written=datetime.now(timezone.utc).isoformat(timespec="seconds"))
    (cfg_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
    for s in range(len(shards)):
        (shard_dir / f"{s:05d}.npz").unlink()
    shard_dir.rmdir()
    run_path.unlink()
    print(f"{cid}: wrote {X.shape} -> {cfg_dir}")
    return cfg_dir


def write_image_index(manifest_path, out_dir, limit=None):
    manifest = pd.read_csv(manifest_path, low_memory=False)
    if limit:
        manifest = manifest.iloc[:limit]
    index = image_index(manifest)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    index.to_parquet(Path(out_dir) / "images.parquet", index=False)
    return index


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--fre", type=Path, required=True, help="an FRE .npz (fre.py)")
    parser.add_argument("--B", type=int, default=11, help="self-description neighbourhood")
    parser.add_argument("--L", type=int, default=3, help="number of scales")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    args = parser.parse_args()
    write_image_index(args.manifest, args.out, args.limit)
    extract(args.manifest, args.fre, args.B, args.L, args.out, args.workers, args.limit)


if __name__ == "__main__":
    main()
