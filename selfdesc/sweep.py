"""Train the FREs a grid needs, then extract self-descriptions for every config in it.

    python -m selfdesc.sweep --manifest experiments/selfdesc-sample/manifest.csv \\
        --out experiments/selfdesc-sample --workers 8

    # server, reusing the FREs frozen on the local sample so the two runs are comparable:
    python -m selfdesc.sweep --manifest experiments/stratified-36k/manifest.csv \\
        --out experiments/selfdesc-36k --fre-dir experiments/selfdesc-sample/fre --workers 40

The default grid varies one thing at a time around the paper's setting (K=8, M=11, B=11, L=3):

    K in {1, 2, 4, 8, 16}      number of residuals = number of filters in Phi
    B in {5, 7, 11}            Phi's neighbourhood
    L in {1, 2, 3}             number of scales
    + the authors' released FRE (K=8, M=15) at B=11, L=3, as an external anchor

Each FRE is trained once on the manifest's authentic `split == "train"` rows (real images only;
no synthetic image or label is ever seen) and frozen with a hash. If --fre-dir already holds a
file for a given (K, M) it is used as is. lambda is calibrated once (fre.calibrate_lambda) and
stored in <fre-dir>/lambda.json; every FRE records the lambda it was trained with.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import extract as ex
from . import fre

PAPER = {"K": 8, "M": 11, "B": 11, "L": 3}
DEFAULT_GRID = (
    [dict(PAPER, K=k) for k in (1, 2, 4, 8, 16)]
    + [dict(PAPER, B=b) for b in (5, 7)]
    + [dict(PAPER, L=l) for l in (1, 2)]
)
UPSTREAM = {"K": 8, "M": 15, "B": 11, "L": 3, "upstream": True}


def training_paths(manifest):
    real = manifest[(manifest["label"] == 0) & (manifest["split"] == "train")]
    assert len(real), "no authentic train rows to learn an FRE from"
    return real["image_id"].tolist(), [ex.PROJECT_ROOT / p for p in real["crop_path"]]


def ensure_lambda(fre_dir, manifest, manifest_path):
    path = fre_dir / "lambda.json"
    if path.exists():
        return json.loads(path.read_text())["lambda"]
    ids, paths = training_paths(manifest)
    R15, windows = fre.second_moments(paths, 15)
    lam, info = fre.calibrate_lambda(R15)
    record = dict(info, **{"lambda": lam, "calibrated_on": "K=8, M=15, our authentic train crops",
                           "target": "fre.UPSTREAM_SINGULAR_VALUES (mean of minor values)",
                           "upstream_sha256": fre.UPSTREAM_SHA256,
                           "training_images": len(ids), "training_windows": windows,
                           "manifest": str(manifest_path)})
    fre_dir.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n")
    print(f"calibrated lambda = {lam:.4f} ({info})")
    return lam


def ensure_fre(fre_dir, K, M, lam, manifest, manifest_path, moments_cache):
    path = fre_dir / f"{fre.fre_name(K, M)}.npz"
    if path.exists():
        return path
    ids, paths = training_paths(manifest)
    if M not in moments_cache:
        moments_cache[M] = fre.second_moments(paths, M)
    R, windows = moments_cache[M]
    filters, info = fre.train(R, M, K, lam=lam)
    assert info["converged"], f"FRE K={K} M={M} did not converge: {info['optimizer_message']}"
    extra = {"trained_on": "authentic rows with split == 'train' (real images only)",
             "training_images": len(ids), "training_windows": windows,
             "training_ids_sha256": hashlib.sha256("\n".join(sorted(ids)).encode()).hexdigest(),
             "manifest": str(manifest_path), "manifest_sha256": ex.file_sha256(manifest_path),
             "pixels": "PIL 'L' luma, 0..255, of the crop_policy crops"}
    print(f"trained {path.name}: energy/filter {np.mean(info['energy_per_filter']):.2f}, "
          f"singular values {np.round(info['singular_values'][:4], 3).tolist()}...")
    return fre.save(filters, info, fre_dir, extra)


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--fre-dir", type=Path, default=None,
                        help="where FREs live / are written (default: <out>/fre)")
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--only", nargs="*", default=None, metavar="CONFIG",
                        help="restrict to these config ids, e.g. K8_M11_B11_L3")
    parser.add_argument("--no-upstream", action="store_true",
                        help="skip the authors' released FRE (needs network to fetch)")
    parser.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    args = parser.parse_args()

    fre_dir = args.fre_dir or args.out / "fre"
    manifest = pd.read_csv(args.manifest, low_memory=False)
    ex.write_image_index(args.manifest, args.out, args.limit)

    grid = DEFAULT_GRID + ([] if args.no_upstream else [UPSTREAM])
    lam = ensure_lambda(fre_dir, manifest, args.manifest)
    moments = {}
    jobs = []
    for cfg in grid:
        if cfg.get("upstream"):
            path = fre_dir / f"fre_upstream_K{cfg['K']}_M{cfg['M']}.npz"
            if not path.exists():
                path = fre.fetch_upstream(fre_dir)
        else:
            path = ensure_fre(fre_dir, cfg["K"], cfg["M"], lam, manifest, args.manifest, moments)
        cid = ex.config_id(path, cfg["B"], cfg["L"])
        if args.only is None or cid in args.only:
            jobs.append((path, cfg["B"], cfg["L"]))

    for path, B, L in jobs:
        ex.extract(args.manifest, path, B, L, args.out, args.workers, args.limit)


if __name__ == "__main__":
    main()
