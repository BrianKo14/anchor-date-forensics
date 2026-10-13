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

__all__ = ["config_id", "image_index", "extract", "file_sha256", "write_image_index"]

# Every image in the sample went through imaging.prepare_image: decoded, alpha-composited onto
# white, re-encoded as JPEG q95. That is the file a crop200_align16 crop was cut from. A
# native_* crop (imports/native_crop_cache.py) skips the re-encode: same decode and composite,
# cut straight from the source bytes.
STORED_FORMAT = f"JPEG q{common.JPEG_QUALITY} (imaging.prepare_image)"
NATIVE_STORED_FORMAT = "source as published, decoded + alpha onto white, no re-encode"


def stored_format(policy):
    return NATIVE_STORED_FORMAT if policy.startswith("native_") else STORED_FORMAT

# The authentic importers do not record a source container. Filled only where the source fixes
# it: COCO ships JPEG; the RAISE importer downloads the TIFF column of RAISE_urls.csv. LAION is
# whatever each scraped URL served, so it stays unknown rather than guessed.
AUTHENTIC_SOURCE_FORMAT = {"COCO2017": "JPEG", "COCO2017_train": "JPEG", "COCO2017_val": "JPEG",
                           "RAISE": "TIFF"}


def config_id(fre_path, B, L, channels="luma"):
    """fre_K8_M11.npz + B=11, L=3 -> "K8_M11_B11_L3"; with channels="rgb", "K8_M11_B11_L3_rgb"."""
    suffix = "" if channels == "luma" else f"_{channels}"
    return f"{Path(fre_path).stem.removeprefix('fre_')}_B{B}_L{L}{suffix}"


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
    index["crop_policy"] = crop_policy(manifest)
    index["stored_format"] = stored_format(index["crop_policy"].iloc[0])
    index["release_date"] = pd.to_datetime(index["release_date"], errors="coerce")
    return index


# --- workers -----------------------------------------------------------------------------------

def _describe_shard(args):
    """Describe one shard's rows and write the shard atomically. Runs in a worker process."""
    shard_path, rows, fre_path, B, L, channels = args
    filters, _ = fre.load(fre_path)
    K = filters.shape[0]
    C = core.CHANNELS[channels]
    X = np.empty((len(rows), C * core.description_dim(K, B)), dtype=np.float32)
    energy = np.empty((len(rows), C * K))
    fit_mse = np.empty(len(rows))
    phi_norm = np.empty(len(rows))
    for i, (_, crop_path) in enumerate(rows):
        # one Phi per plane, channel-major; the diagnostics pool the planes
        fits = [core.describe(plane, filters, B, L)
                for plane in core.load_channels(PROJECT_ROOT / crop_path, channels)]
        X[i] = np.concatenate([phi for phi, _ in fits])
        energy[i] = np.concatenate([diag["residual_energy"] for _, diag in fits])
        fit_mse[i] = np.mean([diag["fit_mse"] for _, diag in fits])
        phi_norm[i] = float(np.sqrt(sum(diag["phi_norm"] ** 2 for _, diag in fits)))
    tmp = shard_path.with_name(f".{shard_path.stem}.{os.getpid()}.tmp.npz")
    np.savez(tmp, X=X, image_id=np.array([image_id for image_id, _ in rows]),
             residual_energy=energy, fit_mse=fit_mse, phi_norm=phi_norm)
    os.replace(tmp, shard_path)
    return len(rows)


class _Run:
    """One config's run: the pinned run.json, its shard layout, and the paths that go with it.

    Shared by the all-in-one `extract` and by the runner-driven `--shard` / `--assemble` modes
    (server-run branch, runner/selfdesc.py), so a description assembled from shards the runner
    computed one subprocess at a time is the same file, byte for byte, as one computed here.
    """

    def __init__(self, manifest_path, fre_path, B, L, out_dir, limit=None, shard_size=None,
                 workers=1, channels="luma"):
        self.manifest_path, self.fre_path = Path(manifest_path), Path(fre_path)
        self.B, self.L, self.channels = B, L, channels
        self.manifest = pd.read_csv(self.manifest_path, low_memory=False)
        if limit:
            self.manifest = self.manifest.iloc[:limit]
        filters, fre_sha = fre.load(self.fre_path)
        self.K, self.M, _ = filters.shape
        self.cid = config_id(self.fre_path, B, L, channels)
        self.cfg_dir = Path(out_dir) / "descriptions" / self.cid
        self.shard_dir = self.cfg_dir / "shards"
        self.n = len(self.manifest)
        if shard_size is None:
            # Enough shards to keep every worker busy, few enough to stay cheap at 355k rows.
            shard_size = int(min(1024, max(16, -(-self.n // (4 * max(workers, 1))))))
        self.shard_size = shard_size
        self.n_shards = -(-self.n // shard_size)
        self.record = {"config": self.cid, "K": self.K, "M": self.M, "B": B, "L": L,
                       "channels": channels,
                       "fre_sha256": fre_sha, "manifest_sha256": file_sha256(self.manifest_path),
                       "limit": limit, "rows": self.n, "shard_size": shard_size,
                       "crop_policy": crop_policy(self.manifest)}

    @property
    def assembled(self):
        # meta.json is written last, so its presence means every other output is complete.
        return (self.cfg_dir / "meta.json").exists()

    def pin(self):
        """Write run.json, or refuse if shards on disk were made under a different run."""
        run_path = self.cfg_dir / "run.json"
        if run_path.exists():
            previous = json.loads(run_path.read_text())
            if previous != self.record:
                diff = {k: (previous.get(k), self.record[k]) for k in self.record
                        if previous.get(k) != self.record[k]}
                raise SystemExit(f"{self.cid}: existing shards were made under a different run "
                                 f"{diff}; delete {self.cfg_dir} to start over")
        self.shard_dir.mkdir(parents=True, exist_ok=True)
        # Atomic: under the runner, many shard subprocesses pin the same config at once, and a
        # reader must never see a half-written file.
        tmp = run_path.with_name(f".run.json.{os.getpid()}.tmp")
        tmp.write_text(json.dumps(self.record, indent=2) + "\n")
        os.replace(tmp, run_path)

    def shard_path(self, s):
        return self.shard_dir / f"{s:05d}.npz"

    def task(self, s):
        start = s * self.shard_size
        part = self.manifest.iloc[start:start + self.shard_size]
        rows = list(zip(part["image_id"], part["crop_path"]))
        return (self.shard_path(s), rows, self.fre_path, self.B, self.L, self.channels)

    def missing(self):
        return [s for s in range(self.n_shards) if not self.shard_path(s).exists()]

    def assemble(self, workers=None, seconds=None, execution="extract.py"):
        """Concatenate the shards into descriptions.npz + diagnostics + meta.json; drop shards."""
        missing = self.missing()
        if missing:
            raise SystemExit(f"{self.cid}: {len(missing)} shard(s) missing, cannot assemble "
                             f"(first: {missing[:5]})")
        shards = [np.load(self.shard_path(s)) for s in range(self.n_shards)]
        X = np.concatenate([s["X"] for s in shards])
        ids = np.concatenate([s["image_id"] for s in shards])
        assert (ids == self.manifest["image_id"].to_numpy()).all(), \
            "shard order drifted from the manifest"
        assert np.isfinite(X).all(), f"{self.cid}: non-finite descriptions"

        diag = pd.DataFrame({"image_id": ids,
                             "fit_mse": np.concatenate([s["fit_mse"] for s in shards]),
                             "phi_norm": np.concatenate([s["phi_norm"] for s in shards])})
        energy = np.concatenate([s["residual_energy"] for s in shards])
        planes = core.CHANNELS[self.channels]
        for j in range(planes * self.K):
            name = (f"residual_energy_{j}" if planes == 1
                    else f"residual_energy_{'rgb'[j // self.K]}{j % self.K}")
            diag[name] = energy[:, j]

        np.savez(self.cfg_dir / "descriptions.npz", X=X, image_id=ids)
        diag.to_parquet(self.cfg_dir / "diagnostics.parquet", index=False)
        fre_meta_path = self.fre_path.with_suffix(".meta.json")
        meta = dict(self.record,
                    dim=int(X.shape[1]),
                    fre_file=_display_path(self.fre_path),
                    fre_meta=(json.loads(fre_meta_path.read_text())
                              if fre_meta_path.exists() else None),
                    manifest=_display_path(self.manifest_path),
                    ridge=core.RIDGE,
                    upstream_reference=("github.com/ductai199x/"
                                        "Forensic-Self-Descriptions-CVPR25@50f2eae"),
                    deviations=["no resize to 1024: input is the native-pixel crop (crop_policy)",
                                "lambda calibrated to the released FRE's spectral diversity "
                                "(fre.py)"]
                    + (["R, G, B described separately with the luma-trained FRE, "
                        "concatenated channel-major (core.load_channels)"]
                       if self.channels != "luma" else []),
                    git_commit=git_commit(),
                    execution=execution,
                    workers=workers,
                    seconds_this_run=round(seconds, 1) if seconds is not None else None,
                    numpy=np.__version__,
                    written=datetime.now(timezone.utc).isoformat(timespec="seconds"))
        (self.cfg_dir / "meta.json").write_text(json.dumps(meta, indent=2) + "\n")
        for s in range(self.n_shards):
            self.shard_path(s).unlink()
        self.shard_dir.rmdir()
        (self.cfg_dir / "run.json").unlink()
        print(f"{self.cid}: wrote {X.shape} -> {self.cfg_dir}")
        return self.cfg_dir


def extract(manifest_path, fre_path, B, L, out_dir, workers=1, limit=None, shard_size=None,
            channels="luma"):
    """Describe every manifest row under one config. Returns the config's output directory."""
    run = _Run(manifest_path, fre_path, B, L, out_dir, limit, shard_size, workers, channels)
    if run.assembled:
        print(f"{run.cid}: already assembled, skipping")
        return run.cfg_dir
    run.pin()

    todo = [run.task(s) for s in run.missing()]
    started = time.time()
    done = run.n - sum(len(t[1]) for t in todo)
    dim = core.CHANNELS[channels] * core.description_dim(run.K, B)
    print(f"{run.cid}: {run.n} rows, D={dim}, {len(todo)} shard(s) "
          f"to do, {done} rows already done")
    if workers > 1 and len(todo) > 1:
        # One BLAS thread per process: the workers already fill the cores, and nested
        # threading on top of them only adds contention. Spawned children read these at import.
        for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                    "VECLIB_MAXIMUM_THREADS"):
            os.environ[var] = "1"
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for count in pool.map(_describe_shard, todo):
                done += count
                print(f"  {done}/{run.n}  ({time.time() - started:.0f}s)", flush=True)
    else:
        for task in todo:
            done += _describe_shard(task)
            print(f"  {done}/{run.n}  ({time.time() - started:.0f}s)", flush=True)
    return run.assemble(workers, time.time() - started)


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
    parser.add_argument("--channels", choices=sorted(core.CHANNELS), default="luma",
                        help="describe the luma plane (upstream) or R, G, B separately")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=os.cpu_count())
    parser.add_argument("--limit", type=int, default=None, help="first N rows only (smoke test)")
    parser.add_argument("--shard-size", type=int, default=None,
                        help="rows per shard; must be the same on every --shard/--assemble call")
    parser.add_argument("--shard", type=int, default=None, metavar="S",
                        help="compute only shard S, single-process, then exit (runner mode)")
    parser.add_argument("--assemble", action="store_true",
                        help="assemble a config whose shards are all on disk, then exit "
                             "(runner mode)")
    args = parser.parse_args()

    if args.shard is not None or args.assemble:
        run = _Run(args.manifest, args.fre, args.B, args.L, args.out, args.limit,
                   args.shard_size, channels=args.channels)
        if run.assembled:
            print(f"{run.cid}: already assembled")
            return
        run.pin()
        if args.shard is not None:
            if not 0 <= args.shard < run.n_shards:
                raise SystemExit(f"shard {args.shard} out of range 0..{run.n_shards - 1}")
            if run.shard_path(args.shard).exists():
                print(f"{run.cid}: shard {args.shard} already on disk")
            else:
                started = time.time()
                count = _describe_shard(run.task(args.shard))
                print(f"{run.cid}: shard {args.shard}: {count} rows in "
                      f"{time.time() - started:.1f}s")
        if args.assemble:
            run.assemble(execution="runner (server-run branch): one subprocess per shard")
        return

    write_image_index(args.manifest, args.out, args.limit)
    extract(args.manifest, args.fre, args.B, args.L, args.out, args.workers, args.limit,
            channels=args.channels)


if __name__ == "__main__":
    main()
