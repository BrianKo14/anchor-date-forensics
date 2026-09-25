"""Build a detector-panel manifest CSV from the imported sample.

Shared across experiments: an experiment is a manifest plus an output directory, so this
script takes filters and an output path rather than hardcoding either.

    python imports/build_manifest.py --out manifest.csv

The emitted schema is the panel's input contract:

    image_id, path, label, generator_family, release_date, generator, crop_path,
    source_width, source_height, crop_top, crop_left, crop_height, crop_width

`path` points at the imported JPEG, `crop_path` at the preprocessed crop the detectors
actually read (see preprocess_crop_cache.py). Both are repo-root-relative, matching the
convention the importers already use.

The last six columns are the crop's provenance: the image's native dimensions and the exact
box taken out of it. They are written here rather than by the cache builder because the
manifest is what every downstream artifact joins on -- the score files, master_scores.csv,
and eventually the attestation emitter's panel manifest. preprocess_crop_cache.py *reads*
these coordinates rather than recomputing them, so the recorded box is by construction the
box that was cut: a crop whose location or native size cannot be recovered later is a hole
in the audit trail, and RAISE is where it would hurt most (a 4928x3264 scan reduced to one
200x200 patch with no record of which patch).
"""

import argparse
import json
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sample"))

import pandas as pd  # noqa: E402

import common  # noqa: E402

MANIFEST_COLUMNS = [
    "image_id",
    "path",
    "label",
    # Which corpus a real image came from, and which side of AI-GenBench's partition any image
    # sits on. Both were previously recoverable only by parsing `path`, which is how
    # master_scores.csv ends up with a derived `source` column. Carried explicitly because the
    # archival false-positive rate is measured on origin_dataset == "RAISE" alone, and a
    # stratified draw deliberately over-represents it -- pooling the authentic class would
    # silently answer a different question.
    "origin_dataset",
    "split",
    "generator_family",
    "release_date",
    "generator",
    "crop_path",
    "source_width",
    "source_height",
    "crop_top",
    "crop_left",
    "crop_height",
    "crop_width",
]

AUTHENTIC_FAMILY = "authentic"

# Coarse family labels for AI-GenBench's 36 generators.
#
# AI-GenBench itself ships only {name: release_date} -- there is no family field anywhere in
# its metadata -- so this map is authored here and is the single source of truth for every
# experiment. Grouping is by *what the artifact looks like* rather than by strict architectural
# lineage, because that is what a per-family score model is conditioning on:
#
#   gan            single forward pass through an adversarially-trained generator
#   diffusion      iterative denoising sampler (incl. latent and rectified-flow variants)
#   autoregressive discrete token prediction over a learned codebook
#   inpainting     only part of the frame is synthesised; the rest is authentic pixels
#   graphics       rendered by a 3D pipeline, not a neural generator at all
#   other          feed-forward neural synthesis that is none of the above
#
# Judgement calls worth knowing about, since they move images between families:
#   * "Diffusion GAN (...)" are GANs whose *discriminator* sees diffusion-noised inputs; the
#     generator is still a one-shot GAN, so the output statistics are GAN-like -> gan.
#   * "Denoising Diffusion GAN" instead samples through a reverse diffusion chain whose
#     denoiser is a GAN -> diffusion.
#   * VQGAN is adversarially trained but its images are composed by an autoregressive
#     transformer over VQ tokens -> autoregressive.
#   * FaceSynthetics is Microsoft's *rendered* face corpus. Detectors trained on GAN or
#     diffusion artifacts have no reason to fire on it, so it must not be pooled with them.
GENERATOR_FAMILIES = {
    "CycleGAN": "gan",
    "Cascaded Refinement Networks": "other",
    "ProGAN": "gan",
    "StarGAN": "gan",
    "SN-PatchGAN": "inpainting",
    "BigGAN": "gan",
    "IMLE": "other",
    "StyleGAN1": "gan",
    "GauGAN": "gan",
    "StyleGAN2": "gan",
    "DDPM": "diffusion",
    "CIPS": "gan",
    "VQGAN": "autoregressive",
    "GANformer": "gan",
    "ADM": "diffusion",
    "StyleGAN3": "gan",
    "LaMa": "inpainting",
    "FaceSynthetics": "graphics",
    "ProjectedGAN": "gan",
    "Palette": "diffusion",
    "VQ-Diffusion": "diffusion",
    "Denoising Diffusion GAN": "diffusion",
    "Glide": "diffusion",
    "Latent Diffusion": "diffusion",
    "Midjourney": "diffusion",
    "MAT": "inpainting",
    "Diffusion GAN (ProjectedGAN)": "gan",
    "Diffusion GAN (StyleGAN2)": "gan",
    "Stable Diffusion 1.4": "diffusion",
    "Stable Diffusion 1.5": "diffusion",
    "Stable Diffusion 2.1": "diffusion",
    "DeepFloyd IF": "diffusion",
    "Stable Diffusion XL 1.0": "diffusion",
    "DALL-E 3": "diffusion",
    "FLUX 1 Dev": "diffusion",
    "FLUX 1 Schnell": "diffusion",
}


def allocate(sizes, total):
    """Split `total` across strata proportionally to `sizes`, summing to exactly `total`.

    Largest-remainder rather than rounding each share independently: rounding leaves the parts
    summing to total +- a few, and a manifest that is 17,998 rows when it claims 18,000 is the
    kind of discrepancy that surfaces three steps downstream as an unexplained imbalance.
    """
    grand = sum(sizes.values())
    if grand == 0:
        return {key: 0 for key in sizes}

    exact = {key: total * size / grand for key, size in sizes.items()}
    floors = {key: min(int(value), sizes[key]) for key, value in exact.items()}
    shortfall = total - sum(floors.values())

    # Hand the remainder to the largest fractional parts, skipping any stratum already exhausted.
    order = sorted(sizes, key=lambda key: (exact[key] - floors[key], key), reverse=True)
    index = 0
    while shortfall > 0 and index < len(order) * 2:
        key = order[index % len(order)]
        if floors[key] < sizes[key]:
            floors[key] += 1
            shortfall -= 1
        index += 1
    return floors


def draw(frame, n, key, seed):
    """`n` rows out of `frame`, preserving the benchmark-split proportions inside it.

    Preserving the split matters because AI-GenBench's train/validation partition is the cut every
    downstream calibration uses. A flat random draw over a 4000/1000 generator would land near 80/20
    but not on it, and the drift is per-generator, so some families would calibrate on noticeably
    less data than others for no reason anyone could reconstruct later.

    Seeded from the stratum's own name rather than one global RNG, so adding or dropping a generator
    leaves every other generator's draw byte-identical -- the same stability common.assign_splits is
    built around, and what makes growing the sample cheap instead of a full reshuffle.
    """
    if n >= len(frame):
        return frame

    sizes = frame["split"].value_counts().to_dict()
    per_split = allocate(sizes, n)

    parts = []
    for split, count in sorted(per_split.items()):
        if count == 0:
            continue
        members = frame[frame["split"] == split].sort_values("file_id")
        order = list(range(len(members)))
        random.Random(f"{seed}:{key}:{split}").shuffle(order)
        parts.append(members.iloc[sorted(order[:count])])
    return pd.concat(parts) if parts else frame.iloc[:0]


def stratified_sample(sample, per_generator, keep_whole, seed):
    """An equal-per-generator draw of fakes plus a label-balanced authentic draw.

    Two deliberate asymmetries:

      * Fakes are equal per generator, not proportional. They are already uniform at 5,000 each in
        the published set, and the per-family score models each need their own calibration data --
        a family is not better calibrated for being more numerous in someone else's benchmark.

      * Authentic is matched to the fake total (the 50/50 construction the likelihood ratio
        depends on: an unbalanced draw smuggles a prevalence prior into what is supposed to be a
        pure LR), but is NOT proportional within itself. `keep_whole` sources are taken entirely.
        RAISE is the reason: it is the only genuinely digitized archival material here and the
        false-positive rate on it is the thesis's central open question, so all 1,000 images go in
        and the archival arm gets the tightest interval the data allows. That deliberately
        over-represents RAISE relative to the benchmark's own composition -- downstream analysis
        must split by origin_dataset rather than pooling the authentic class.
    """
    fakes = sample[sample["label"] == common.FAKE_LABEL]
    authentic = sample[sample["label"] == common.REAL_LABEL]

    short = {g: len(f) for g, f in fakes.groupby("generator") if len(f) < per_generator}
    if short:
        raise SystemExit(
            f"--stratify {per_generator} exceeds what these generators have: "
            + ", ".join(f"{g}={n}" for g, n in sorted(short.items()))
        )

    drawn_fakes = pd.concat(
        [draw(frame, per_generator, generator, seed)
         for generator, frame in sorted(fakes.groupby("generator"))]
    )

    target = len(drawn_fakes)
    whole = authentic[authentic["origin_dataset"].isin(keep_whole)]
    rest = authentic[~authentic["origin_dataset"].isin(keep_whole)]
    remaining = target - len(whole)
    if remaining < 0:
        raise SystemExit(
            f"--keep-whole sources supply {len(whole)} authentic images, more than the "
            f"{target} needed to match the fakes; lower --keep-whole or raise --stratify"
        )

    sizes = {origin: len(frame) for origin, frame in rest.groupby("origin_dataset")}
    per_origin = allocate(sizes, remaining)
    drawn_authentic = pd.concat(
        [whole]
        + [draw(frame, per_origin[origin], origin, seed)
           for origin, frame in sorted(rest.groupby("origin_dataset")) if per_origin[origin]]
    )

    return pd.concat([drawn_authentic, drawn_fakes]).reset_index(drop=True)


def write_stratify_meta(args, manifest):
    """Record how a stratified manifest was drawn, beside the manifest itself.

    The draw is reproducible from (data root, seed, per-generator N, keep-whole) alone, but only
    if those four are written down. Everything downstream -- the score files, the calibration, the
    attestation emitter's panel manifest -- chains to this CSV, and "which subset of 355,638 was
    this?" is not a question a bare CSV of 36,000 rows can answer about itself.
    """
    def composition(frame, column):
        return {str(key): int(value) for key, value in frame[column].value_counts().sort_index().items()}

    fakes = manifest[manifest["label"] == common.FAKE_LABEL]
    authentic = manifest[manifest["label"] == common.REAL_LABEL]

    meta = {
        "stratification": {
            "per_generator": args.stratify,
            "keep_whole": list(args.keep_whole),
            "seed": args.seed,
            "benchmark_splits": ["train", "validation"] if args.include_validation else [common.BENCHMARK_SPLIT],
        },
        "data_root": str(common.DATA_DIR),
        "crop_policy": crop_policy_id(args.crop_size, args.crop_align),
        "rows": len(manifest),
        "labels": {"authentic": len(authentic), "fake": len(fakes)},
        "by_split": composition(manifest, "split"),
        "authentic_by_origin": composition(authentic, "origin_dataset"),
        "fakes_by_generator": composition(fakes, "generator"),
        "fakes_by_family": composition(fakes, "generator_family"),
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
    path = Path(str(args.out).removesuffix(".csv") + ".meta.json")
    path.write_text(json.dumps(meta, indent=2) + "\n")
    return path


def crop_policy_id(size, align):
    """Identify a crop policy so caches for different policies cannot collide."""
    return f"crop{size}_align{align}"


def crop_dir(policy_id):
    return common.DATA_DIR / "preprocessed" / policy_id


def crop_path(file_id, policy_id):
    """Repo-root-relative path to a file_id's cached crop. Mirrors common.image_path's flattening."""
    return f"data/preprocessed/{policy_id}/{file_id.replace('/', '_')}.png"


def crop_box(width, height, size, align):
    """Centre crop of `size`, origin snapped down to a multiple of `align`.

    The centring itself is a port of AI-GenBench's RandomCropIfLarge.get_crop_params
    (training_and_evaluation/lightning_data_modules/augmentation_utils/random_crop_if_large.py,
    the force_central_crop=True branch: crop_width/height = min(side, threshold), origin =
    (side - crop_side) // 2), not a reimplementation from scratch -- same convention imaging.py
    uses for AI-GenBench's prepare_image. Importing the class itself would pull torch and
    torchvision into a venv that otherwise needs neither, plus the rest of that package's
    __init__ chain (pytorch_lightning, albumentations), for six lines of arithmetic.

    Two things this project adds on top of the ported formula:
      * the align-down-to-16 step, which AI-GenBench's version does not do -- see
        preprocess_crop_cache.py for why (JPEG block-grid preservation).
      * raising on an under-size image rather than AI-GenBench's min(side, threshold), which
        would silently return a *smaller-than-size* crop instead. Doesn't currently trigger --
        imaging.decode_and_validate already rejects anything under common.IMAGE_MIN_SIZE == 200,
        so every crop here is a full 200x200 -- but a silent short crop is exactly the kind of
        thing that belongs in the manifest's audit trail if the size floor ever changes.

    Returns (top, left, height, width) -- torchvision's crop-parameter order, matching what
    get_crop_params itself returns.

    Nothing is ever upscaled or resampled -- this policy only ever removes pixels, which is
    what keeps RAISE's 4928x3264 scans out of the interpolation that a resize would impose.
    """
    if width < size or height < size:
        raise ValueError(f"image is {width}x{height}, smaller than the {size}x{size} crop")
    crop_width = min(width, size)
    crop_height = min(height, size)
    left = ((width - crop_width) // 2 // align) * align
    top = ((height - crop_height) // 2 // align) * align
    return top, left, crop_height, crop_width


def family_of(row):
    generator = (row["generator"] or "").strip()
    if row["label"] == common.REAL_LABEL:
        return AUTHENTIC_FAMILY
    return GENERATOR_FAMILIES[generator]


def build(size, align, split=None, origin_dataset=None, label=None, benchmark_splits=None,
          stratify=None, keep_whole=(), seed=common.SPLIT_SEED):
    """The sample as a panel manifest DataFrame, filtered as requested."""
    policy_id = crop_policy_id(size, align)
    sample = common.load_sample(splits=benchmark_splits)

    if split is not None:
        sample = sample[sample["split"] == split]
    if origin_dataset is not None:
        sample = sample[sample["origin_dataset"] == origin_dataset]
    if label is not None:
        sample = sample[sample["label"] == label]
    if sample.empty:
        raise SystemExit("no rows left after filtering -- check --split/--origin-dataset/--label")

    # After the filters, so --stratify composes with them rather than being silently overridden.
    if stratify is not None:
        sample = stratified_sample(sample, stratify, keep_whole, seed)

    # Fail loudly on an unmapped generator rather than silently emitting a blank family: a
    # missing family would quietly pool a new generator with nothing, and the per-family score
    # models downstream would never notice.
    fakes = sample[sample["label"] == common.FAKE_LABEL]
    unmapped = sorted(set(fakes["generator"]) - set(GENERATOR_FAMILIES))
    if unmapped:
        raise SystemExit(
            "generators missing from GENERATOR_FAMILIES in imports/build_manifest.py: "
            + ", ".join(unmapped)
        )

    manifest = sample.copy()
    manifest["image_id"] = manifest["file_id"]
    manifest["generator_family"] = manifest.apply(family_of, axis=1)
    manifest["crop_path"] = [crop_path(fid, policy_id) for fid in manifest["file_id"]]

    # Crop provenance. The importers recorded each image's native size; the box is derived
    # from it here so that the manifest, not the cache builder, is the single place the crop
    # geometry is decided. An under-size image is fatal and names itself -- silently emitting
    # a short crop would hand the panel a differently-shaped input with no trace in the CSV.
    boxes = []
    for row in manifest.itertuples():
        try:
            boxes.append(crop_box(int(row.width), int(row.height), size, align))
        except ValueError as error:
            raise SystemExit(f"{row.file_id}: {error}")
    manifest["source_width"] = manifest["width"].astype(int)
    manifest["source_height"] = manifest["height"].astype(int)
    for column, values in zip(("crop_top", "crop_left", "crop_height", "crop_width"), zip(*boxes)):
        manifest[column] = list(values)

    if "release_date" not in manifest.columns:
        manifest["release_date"] = ""
    manifest["release_date"] = manifest["release_date"].fillna("")
    manifest["generator"] = manifest["generator"].fillna("")

    # image_id flattening ("/" -> "_") could in principle collide, e.g. "a/b" and "a_b". The
    # sample importers share this scheme so a collision would already have overwritten an
    # image on disk, but assert it here rather than trust that: a collision downstream is a
    # crop silently scored twice under one id.
    if manifest["crop_path"].duplicated().any():
        clashes = manifest.loc[manifest["crop_path"].duplicated(keep=False), ["image_id", "crop_path"]]
        raise SystemExit(f"crop_path collision between flattened image_ids:\n{clashes}")
    assert manifest["image_id"].is_unique, "image_id is not unique"

    return manifest[MANIFEST_COLUMNS].reset_index(drop=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=common.PROJECT_ROOT / "manifest.csv")
    parser.add_argument("--crop-size", type=int, default=200)
    parser.add_argument("--crop-align", type=int, default=16)
    parser.add_argument("--split", choices=[common.TRAIN_SPLIT, common.VAL_SPLIT], default=None)
    parser.add_argument("--origin-dataset", default=None, help="e.g. RAISE, COCO2017, LAION-400M")
    parser.add_argument("--label", type=int, choices=[common.REAL_LABEL, common.FAKE_LABEL], default=None)
    parser.add_argument("--include-validation", action="store_true",
                         help="also load AI-GenBench's validation partition, not just "
                              "common.BENCHMARK_SPLIT (train) -- only meaningful once both "
                              "partitions have been imported, e.g. the full-dataset build")
    parser.add_argument("--stratify", type=int, metavar="N",
                        help="take N fakes per generator plus a label-matched authentic draw, "
                             "instead of every row. The full set is 355,638 images; a few hundred "
                             "per generator calibrates a per-family score model just as well and "
                             "costs a tenth of the GPU time")
    parser.add_argument("--keep-whole", nargs="*", default=["RAISE"], metavar="ORIGIN",
                        help="origin datasets taken entirely rather than subsampled "
                             "(default: RAISE, the archival false-positive arm)")
    parser.add_argument("--seed", type=int, default=common.SPLIT_SEED)
    args = parser.parse_args()

    if not common.sample_exists():
        raise SystemExit("sample not imported; run imports/sample/authentic.py and fakes.py first")

    manifest = build(
        args.crop_size,
        args.crop_align,
        split=args.split,
        origin_dataset=args.origin_dataset,
        label=args.label,
        benchmark_splits=("train", "validation") if args.include_validation else None,
        stratify=args.stratify,
        keep_whole=tuple(args.keep_whole),
        seed=args.seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    counts = manifest["generator_family"].value_counts()
    print(f"wrote {len(manifest)} rows -> {args.out}")
    print(f"crop policy: {crop_policy_id(args.crop_size, args.crop_align)}")
    print(f"labels: {dict(manifest['label'].value_counts())}")
    print("families: " + ", ".join(f"{fam}={n}" for fam, n in counts.items()))

    if args.stratify is not None:
        sidecar = write_stratify_meta(args, manifest)
        print(f"sidecar: {sidecar}")


if __name__ == "__main__":
    main()
