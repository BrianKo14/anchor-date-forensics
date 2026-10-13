"""Native-pixel crops of the fakes: the same crop box, cut from the decoded *source* bytes.

    python imports/native_crop_cache.py --manifest experiments/stratified-36k/manifest.csv \\
        --per-generator 200 \\
        --q95-out experiments/selfdesc-arms-q95/manifest.csv \\
        --native-out experiments/selfdesc-arms-native/manifest.csv --workers 12

What the panel's crop200_align16 cache holds is a crop of the file `imaging.prepare_image`
wrote: the source decoded, alpha composited onto white, then **re-encoded as JPEG q95**. That
re-encode is right for the detector panel (both classes share one quantization table, see
imaging.py), but it puts a common compression stage on every fake, which is exactly the kind of
fine pixel correlation a self-description models. This script builds the matching crop
*without* that stage, for the pipeline arm of the self-description experiment (DECISIONS.md
D-011):

  * The source is the published AI-GenBench parquet shard the importer decoded
    (`$AIGENBENCH_DATA_ROOT/cache/parquet/`, fakes only; the authentic half has no retained
    source). Decoding and the alpha composite are prepare_image's, minus the final save, so
    the two crops differ only by that one JPEG generation.
  * Same crop box (the manifest's crop_top/left/height/width), same 16-px alignment, so a
    source that is itself JPEG keeps its own block grid. Nothing is resized.
  * A source that arrived as JPEG (all of ArtiFact, SFHQ-T2I, ...) stays JPEG: its own
    compression is part of the source. Only the extra generation is removed.
  * RGB PNG, lossless, under preprocessed/native_crop200_align16/ beside the panel's cache.

Selection: `--per-generator N` fake rows per generator (seeded), all 36 generators, no
authentic rows (the temporal tests use fakes only, and the FREs are the frozen local ones). Two
manifests come out with the same rows in the same order: one pointing at the existing q95 crops,
one at the native crops. They differ only in crop_path, so the arms compare like with like.

QC, written next to the native manifest as native_crops.parquet: per row, the PIL format and
mode of the source, and the PSNR of the native crop against the q95 crop (luma). A q95
re-encode of the same pixels lands near 40-50 dB; a low value means the row was matched to the
wrong source image, and the script refuses to write the manifest if any row is below 30 dB.
"""

import argparse
import glob
import io
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "sample"))

import common  # noqa: E402

POLICY = "crop200_align16"
NATIVE_POLICY = f"native_{POLICY}"
MIN_PSNR = 30.0
SEED = 20261013


def select(manifest, per_generator, seed=SEED):
    """`per_generator` fake rows of every generator, drawn with a fixed seed, in manifest order."""
    fakes = manifest[manifest["label"] == common.FAKE_LABEL]
    short = fakes.groupby("generator").size()
    assert (short >= per_generator).all(), \
        f"generators with fewer than {per_generator} rows: {short[short < per_generator].to_dict()}"
    picked = fakes.groupby("generator", group_keys=False).sample(per_generator, random_state=seed)
    return picked.sort_index()


def native_crop_path(crop_path):
    parts = Path(crop_path).parts
    assert POLICY in parts, f"{crop_path} is not a {POLICY} crop"
    return str(Path(*[NATIVE_POLICY if p == POLICY else p for p in parts]))


def shard_index(data_dir):
    """{file_id: shard parquet path}, from the per-shard metadata sidecars the importer wrote."""
    out = {}
    for meta in sorted(glob.glob(str(data_dir / "cache" / "fake_meta" / "*.meta.parquet"))):
        shard = data_dir / "cache" / "parquet" / Path(meta).name.replace(".meta.parquet", ".parquet")
        for file_id in pd.read_parquet(meta, columns=["file_id"])["file_id"]:
            out[file_id] = shard
    return out


def decode_like_prepare_image(raw):
    """imaging.prepare_image up to (not including) its JPEG save. Returns (RGB image, format, mode)."""
    image = Image.open(io.BytesIO(raw))
    image.load()
    fmt, mode = image.format, image.mode
    if image.mode != "RGB":
        if image.mode != "RGBA":
            image = image.convert("RGBA")
        background = Image.new("RGBA", image.size, (255, 255, 255))
        image = Image.alpha_composite(background, image).convert("RGB")
    return image, fmt, mode


def _psnr(a, b):
    mse = float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))
    return float("inf") if mse == 0 else 10 * np.log10(255.0 ** 2 / mse)


def _crop_shard(args):
    """Every wanted row of one parquet shard. Runs in a worker; returns QC rows."""
    shard, rows = args
    wanted = {r["image_id"]: r for r in rows}
    out = []
    reader = pq.ParquetFile(shard)
    for batch in reader.iter_batches(columns=["file_id", "image"], batch_size=256):
        ids = batch.column("file_id").to_pylist()
        hits = [i for i, fid in enumerate(ids) if fid in wanted]
        if not hits:
            continue
        images = batch.column("image")
        for i in hits:
            row = wanted.pop(ids[i])
            image, fmt, mode = decode_like_prepare_image(images[i]["bytes"].as_py())
            if image.size != (row["source_width"], row["source_height"]):
                raise SystemExit(f"{row['image_id']}: source is {image.size}, manifest records "
                                 f"{(row['source_width'], row['source_height'])}")
            top, left = row["crop_top"], row["crop_left"]
            crop = image.crop((left, top, left + row["crop_width"], top + row["crop_height"]))
            dest = common.PROJECT_ROOT / row["native_crop_path"]
            dest.parent.mkdir(parents=True, exist_ok=True)
            tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
            crop.save(tmp, format="PNG")
            os.replace(tmp, dest)
            with Image.open(common.PROJECT_ROOT / row["crop_path"]) as q95:
                psnr = _psnr(np.asarray(crop.convert("L")), np.asarray(q95.convert("L")))
            out.append({"image_id": row["image_id"], "native_format": fmt, "native_mode": mode,
                        "psnr_vs_q95": psnr})
    if wanted:
        raise SystemExit(f"{shard}: {len(wanted)} wanted row(s) not found, e.g. {next(iter(wanted))}")
    return out


def build(manifest, workers=1, data_dir=common.DATA_DIR):
    """Write every native crop for `manifest`'s rows; return the QC table (manifest order)."""
    where = shard_index(data_dir)
    missing = [i for i in manifest["image_id"] if i not in where]
    assert not missing, f"{len(missing)} row(s) in no shard sidecar, e.g. {missing[:3]}"
    rows = manifest.assign(native_crop_path=manifest["crop_path"].map(native_crop_path))
    by_shard = {}
    for record in rows.to_dict("records"):
        by_shard.setdefault(where[record["image_id"]], []).append(record)
    tasks = sorted(by_shard.items(), key=lambda t: -len(t[1]))
    qc = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for n, part in enumerate(pool.map(_crop_shard, tasks), 1):
                qc += part
                print(f"  {n}/{len(tasks)} shards, {len(qc)}/{len(rows)} crops", flush=True)
    else:
        for n, task in enumerate(tasks, 1):
            qc += _crop_shard(task)
            print(f"  {n}/{len(tasks)} shards, {len(qc)}/{len(rows)} crops", flush=True)
    qc = pd.DataFrame(qc).set_index("image_id").loc[manifest["image_id"]].reset_index()
    return qc


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, required=True,
                        help="a crop200_align16 manifest (build_manifest.py)")
    parser.add_argument("--per-generator", type=int, required=True)
    parser.add_argument("--q95-out", type=Path, required=True)
    parser.add_argument("--native-out", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest, low_memory=False)
    picked = select(manifest, args.per_generator, args.seed)
    print(f"{len(picked)} fake rows, {picked.generator.nunique()} generators, "
          f"{args.per_generator} each (seed {args.seed})")
    qc = build(picked, args.workers)
    low = qc[qc["psnr_vs_q95"] < MIN_PSNR]
    if len(low):
        raise SystemExit(f"{len(low)} crop(s) below {MIN_PSNR} dB against their q95 crop -- "
                         f"source matched to the wrong image? e.g.\n{low.head()}")
    native = picked.assign(crop_path=picked["crop_path"].map(native_crop_path))
    for path, frame in ((args.q95_out, picked), (args.native_out, native)):
        path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_csv(path, index=False)
        print(f"wrote {path}")
    qc.to_parquet(args.native_out.parent / "native_crops.parquet", index=False)
    shown = qc.assign(psnr_vs_q95=qc["psnr_vs_q95"].clip(upper=99.0))   # identical pixels: inf
    print(shown.groupby("native_format").psnr_vs_q95.describe().round(1).to_string())


if __name__ == "__main__":
    main()
