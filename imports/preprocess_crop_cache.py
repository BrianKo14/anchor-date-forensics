"""Build the shared crop cache the detector panel reads.

    python imports/preprocess_crop_cache.py --manifest manifest.csv

Why a shared cache rather than each detector's own preprocessing:

  * The sample's short sides run 200 -> 3264. UniversalFakeDetect's upstream CenterCrop(224)
    would zero-pad 277 fakes and only 19 reals, making padding a class cue -- precisely the
    kind of confound this project exists to avoid.
  * DMimageDetection's res50stride1 does not downsample in its stem, so a 4928x3264 RAISE scan
    produces a ~4 GB activation. Three Stable Diffusion VAEs on the same image fare no better.
  * A likelihood ratio built on a panel needs every member to have seen the *same* pixels.

So: one 200x200 centre crop per image, written once, read by all four detectors.

  * 200 is the sample's exact short-side floor (common.IMAGE_MIN_SIZE) and a multiple of 8, so
    it fits every image with no padding and no resampling, and is a legal SD VAE input.
  * The crop origin is snapped *down* to a multiple of `align` (16) so the JPEG MCU/block grid
    survives the crop. The sample is normalised to JPEG q95 by imaging.prepare_image; an
    unaligned crop would shift the block grid and perturb exactly the artifacts these
    detectors key on.
  * Output is lossless PNG, so no second compression generation is introduced.
  * Nothing is resized. This step only ever removes pixels, so a RAISE scan reaches the panel
    as native 200x200 pixels rather than as an interpolated reduction of the whole frame.

The cache is keyed by policy id (crop200_align16), not a bare directory: the degradation-ladder
experiment will want different pixels from the same sources, and a shared directory would
silently serve the wrong ones.

The crop *geometry* is not decided here. build_manifest.py writes crop_top/crop_left/
crop_height/crop_width and the image's native source_width/source_height into the manifest,
and this script applies exactly those numbers -- so the audit trail records the box that was
actually cut rather than a box recomputed alongside it, and the manifest is the one place the
policy can be changed. The source image is re-measured on the way past and a disagreement with
the recorded native size is fatal: it means the manifest and the images on disk have drifted
apart, and every crop coordinate in the CSV is then describing a different image.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "sample"))

import common  # noqa: E402


GEOMETRY_COLUMNS = [
    "source_width",
    "source_height",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
]


def is_cached(path, size):
    """True when `path` already holds a readable crop of the right (width, height)."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            return image.size == size
    except Exception:
        return False


def build_cache(manifest, force=False):
    written = skipped = 0
    for row in manifest.itertuples():
        source = common.PROJECT_ROOT / row.path
        dest = common.PROJECT_ROOT / row.crop_path
        top, left = int(row.crop_top), int(row.crop_left)
        height, width = int(row.crop_height), int(row.crop_width)

        if not force and is_cached(dest, (width, height)):
            skipped += 1
            continue
        if not source.exists():
            raise SystemExit(f"source image missing: {source}\nre-run the sample importers.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            if image.size != (int(row.source_width), int(row.source_height)):
                raise SystemExit(
                    f"{row.image_id}: manifest records {row.source_width}x{row.source_height} "
                    f"but {source} is {image.width}x{image.height}; the manifest is stale, "
                    f"rebuild it with imports/build_manifest.py"
                )
            if left + width > image.width or top + height > image.height:
                raise SystemExit(
                    f"{row.image_id}: crop box (top={top}, left={left}, height={height}, "
                    f"width={width}) does not fit in {image.width}x{image.height}"
                )
            image = image.convert("RGB")
            image.crop((left, top, left + width, top + height)).save(dest, format="PNG")
        written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=common.PROJECT_ROOT / "manifest.csv")
    parser.add_argument("--force", action="store_true", help="rewrite crops that are already cached")
    args = parser.parse_args()

    # No --size/--align here: the geometry comes from the manifest. They used to be accepted,
    # which meant `--size 224` would quietly fill the crop200_align16 directory with 224px
    # crops that every later run then treated as cached.
    manifest = pd.read_csv(args.manifest)
    missing = [c for c in ["crop_path", *GEOMETRY_COLUMNS] if c not in manifest.columns]
    if missing:
        raise SystemExit(
            f"{args.manifest} is missing {', '.join(missing)}; "
            f"rebuild it with imports/build_manifest.py"
        )

    written, skipped = build_cache(manifest, force=args.force)

    sizes = sorted({(int(w), int(h)) for w, h in zip(manifest["crop_width"], manifest["crop_height"])})
    out_dirs = sorted({str(Path(p).parent) for p in manifest["crop_path"]})
    print(f"crop cache: {written} written, {skipped} skipped ({len(manifest)} rows)")
    print("crop sizes: " + ", ".join(f"{w}x{h}" for w, h in sizes))
    for out_dir in out_dirs:
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
