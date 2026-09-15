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
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
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


class ManifestMismatch(Exception):
    """The manifest's recorded geometry no longer matches the image on disk."""


def crop_one(source, dest, source_width, source_height, crop_top, crop_left, crop_height, crop_width):
    """Cut one manifest row's crop box out of `source` and write it atomically to `dest`.

    Raises FileNotFoundError if `source` is missing, ManifestMismatch if the on-disk image's size
    disagrees with source_width/source_height or the crop box doesn't fit -- both mean the
    manifest and the images on disk have drifted apart. Afterward `dest` either doesn't exist or
    is a complete, valid PNG -- never a partial one: the crop is saved to a temp file in the same
    directory and `os.replace`d into place, so a killed process cannot leave behind a file a later
    resume check mistakes for done. Returns the number of bytes written.
    """
    if not source.exists():
        raise FileNotFoundError(source)

    with Image.open(source) as image:
        if image.size != (source_width, source_height):
            raise ManifestMismatch(
                f"manifest records {source_width}x{source_height} but {source} is "
                f"{image.width}x{image.height}; the manifest is stale, rebuild it with "
                f"imports/build_manifest.py"
            )
        if crop_left + crop_width > image.width or crop_top + crop_height > image.height:
            raise ManifestMismatch(
                f"crop box (top={crop_top}, left={crop_left}, height={crop_height}, "
                f"width={crop_width}) does not fit in {image.width}x{image.height}"
            )
        cropped = image.convert("RGB").crop(
            (crop_left, crop_top, crop_left + crop_width, crop_top + crop_height)
        )

    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(f".{dest.name}.{os.getpid()}.tmp")
    cropped.save(tmp, format="PNG")
    size = tmp.stat().st_size
    os.replace(tmp, dest)
    return size


def _crop_row(args):
    """Picklable per-row worker for the process pool: crop one row, catching its own errors.

    Returns (image_id, error_message_or_None) rather than raising, since a pool worker's
    exception would otherwise just abort that one task silently -- the caller decides what to do
    with a batch of (id, error) results once every row has had a chance to run.
    """
    image_id, path, crop_path, source_width, source_height, top, left, height, width = args
    source = common.resolve_source_path(path)
    dest = common.PROJECT_ROOT / crop_path
    try:
        crop_one(source, dest, source_width, source_height, top, left, height, width)
    except FileNotFoundError:
        return image_id, f"source image missing: {source}"
    except ManifestMismatch as error:
        return image_id, str(error)
    return image_id, None


def build_cache(manifest, force=False, workers=1):
    """Crop every row not already cached. `workers` > 1 fans the work out across processes.

    Sequential and parallel paths share one function (_crop_row) rather than duplicating the
    crop logic per branch -- the only difference is whether `map` runs in this process or a pool.
    """
    skipped = 0
    todo = []
    for row in manifest.itertuples():
        dest = common.PROJECT_ROOT / row.crop_path
        height, width = int(row.crop_height), int(row.crop_width)
        if not force and is_cached(dest, (width, height)):
            skipped += 1
            continue
        todo.append((row.image_id, row.path, row.crop_path, int(row.source_width),
                     int(row.source_height), int(row.crop_top), int(row.crop_left), height, width))

    errors = []
    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for image_id, error in pool.map(_crop_row, todo, chunksize=32):
                if error is not None:
                    errors.append((image_id, error))
    else:
        for args in todo:
            image_id, error = _crop_row(args)
            if error is not None:
                errors.append((image_id, error))

    if errors:
        lines = "\n".join(f"  {image_id}: {error}" for image_id, error in errors[:10])
        more = f"\n  ... and {len(errors) - 10} more" if len(errors) > 10 else ""
        raise SystemExit(f"{len(errors)} row(s) failed:\n{lines}{more}")

    return len(todo), skipped


def _origin_of(row):
    """A per-row label for grouping benchmark timings.

    origin_dataset isn't a manifest column (build_manifest.py doesn't carry it through), but for
    authentic rows it's recoverable from the image_id prefix (e.g. "COCO2017_train/100000" ->
    "COCO2017_train"); for fakes, generator_family is already the right granularity.
    """
    if int(row.label) == common.REAL_LABEL:
        return str(row.image_id).split("/", 1)[0]
    return str(row.generator_family)


def benchmark_cache(manifest, force=False):
    """Time crop_one() per row, grouped by origin, and print an images/s table.

    Reuses the exact production path (crop_one(), common.resolve_source_path()) rather than a
    separate timing harness, so the numbers describe the real job, not an approximation of it.
    Pass --force so cached rows aren't silently skipped -- a benchmark that skips most of its
    sample measures is_cached(), not the crop.
    """
    samples, written_bytes, errors = {}, {}, []

    for row in manifest.itertuples():
        source = common.resolve_source_path(row.path)
        dest = common.PROJECT_ROOT / row.crop_path
        top, left = int(row.crop_top), int(row.crop_left)
        height, width = int(row.crop_height), int(row.crop_width)
        origin = _origin_of(row)

        if not force and is_cached(dest, (width, height)):
            continue

        started = time.perf_counter()
        try:
            size = crop_one(source, dest, int(row.source_width), int(row.source_height),
                             top, left, height, width)
        except FileNotFoundError:
            errors.append(f"{row.image_id}: source missing ({source})")
            continue
        except ManifestMismatch as error:
            errors.append(f"{row.image_id}: {error}")
            continue
        elapsed = time.perf_counter() - started

        samples.setdefault(origin, []).append(elapsed)
        written_bytes[origin] = written_bytes.get(origin, 0) + size

    total_n = sum(len(v) for v in samples.values())
    total_s = sum(sum(v) for v in samples.values())
    print(f"\nbenchmark: {total_n} images, {total_s:.1f}s total, "
          f"{(total_n / total_s if total_s else 0):.1f} images/s overall")
    if errors:
        print(f"  {len(errors)} error(s) skipped, first: {errors[0]}")
    print(f"{'origin':<20} {'n':>6} {'images/s':>10} {'p50 ms':>8} {'p95 ms':>8} {'MB':>8}")
    for origin in sorted(samples):
        times = sorted(samples[origin])
        n, s = len(times), sum(times)
        p50 = times[int(0.50 * (n - 1))] * 1000
        p95 = times[int(0.95 * (n - 1))] * 1000
        print(f"{origin:<20} {n:>6} {(n / s if s else 0):>10.1f} {p50:>8.1f} {p95:>8.1f} "
              f"{written_bytes[origin] / 1e6:>8.1f}")

    return total_n, total_s


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--manifest", type=Path, default=common.PROJECT_ROOT / "manifest.csv")
    parser.add_argument("--force", action="store_true", help="rewrite crops that are already cached")
    parser.add_argument("--benchmark", action="store_true",
                         help="time crop_one() per row, grouped by origin, instead of building the cache")
    parser.add_argument("--workers", type=int, default=1,
                         help="parallel worker processes (default: 1, sequential)")
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

    if args.benchmark:
        benchmark_cache(manifest, force=args.force)
        return

    written, skipped = build_cache(manifest, force=args.force, workers=args.workers)

    sizes = sorted({(int(w), int(h)) for w, h in zip(manifest["crop_width"], manifest["crop_height"])})
    out_dirs = sorted({str(Path(p).parent) for p in manifest["crop_path"]})
    print(f"crop cache: {written} written, {skipped} skipped ({len(manifest)} rows)")
    print("crop sizes: " + ", ".join(f"{w}x{h}" for w, h in sizes))
    for out_dir in out_dirs:
        print(f"  -> {out_dir}")


if __name__ == "__main__":
    main()
