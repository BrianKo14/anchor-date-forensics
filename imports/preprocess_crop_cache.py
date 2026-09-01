"""Build the shared crop cache the detector panel reads.

    python imports/preprocess_crop_cache.py --manifest manifest.csv

Why a shared cache rather than each detector's own preprocessing:

  * The sample's short sides run 200 -> 3264. UniversalFakeDetect's upstream CenterCrop(224)
    would zero-pad 66 fakes and only 4 reals, making padding a class cue -- precisely the kind
    of confound this project exists to avoid.
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

The cache is keyed by policy id (crop200_align16), not a bare directory: the degradation-ladder
experiment will want different pixels from the same sources, and a shared directory would
silently serve the wrong ones.
"""

import argparse
import sys
from pathlib import Path

import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "sample"))

import common  # noqa: E402


def crop_box(width, height, size, align):
    """Centre crop of `size`, with the origin snapped down to a multiple of `align`."""
    if width < size or height < size:
        raise ValueError(f"image is {width}x{height}, smaller than the {size}x{size} crop")
    left = ((width - size) // 2 // align) * align
    top = ((height - size) // 2 // align) * align
    return (left, top, left + size, top + size)


def is_cached(path, size):
    """True when `path` already holds a readable crop of the right size."""
    if not path.exists():
        return False
    try:
        with Image.open(path) as probe:
            probe.verify()
        with Image.open(path) as image:
            image.load()
            return image.size == (size, size)
    except Exception:
        return False


def build_cache(manifest, size, align, force=False):
    written = skipped = 0
    for row in manifest.itertuples():
        source = common.PROJECT_ROOT / row.path
        dest = common.PROJECT_ROOT / row.crop_path

        if not force and is_cached(dest, size):
            skipped += 1
            continue
        if not source.exists():
            raise SystemExit(f"source image missing: {source}\nre-run the sample importers.")

        dest.parent.mkdir(parents=True, exist_ok=True)
        with Image.open(source) as image:
            image = image.convert("RGB")
            image.crop(crop_box(image.width, image.height, size, align)).save(dest, format="PNG")
        written += 1

    return written, skipped


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=common.PROJECT_ROOT / "manifest.csv")
    parser.add_argument("--size", type=int, default=200)
    parser.add_argument("--align", type=int, default=16)
    parser.add_argument("--force", action="store_true", help="rewrite crops that are already cached")
    args = parser.parse_args()

    manifest = pd.read_csv(args.manifest)
    if "crop_path" not in manifest.columns:
        raise SystemExit(f"{args.manifest} has no crop_path column; rebuild it with imports/build_manifest.py")

    written, skipped = build_cache(manifest, args.size, args.align, force=args.force)

    out_dirs = sorted({str(Path(p).parent) for p in manifest["crop_path"]})
    print(f"crop cache: {written} written, {skipped} skipped ({len(manifest)} rows)")
    for out_dir in out_dirs:
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
