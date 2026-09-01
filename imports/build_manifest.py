"""Build a detector-panel manifest CSV from the imported sample.

Shared across experiments: an experiment is a manifest plus an output directory, so this
script takes filters and an output path rather than hardcoding either.

    python imports/build_manifest.py --out manifest.csv

The emitted schema is the panel's input contract:

    image_id, path, label, generator_family, release_date, generator, crop_path

`path` points at the imported JPEG, `crop_path` at the preprocessed crop the detectors
actually read (see preprocess_crop_cache.py). Both are repo-root-relative, matching the
convention the importers already use.
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "sample"))

import common  # noqa: E402

MANIFEST_COLUMNS = [
    "image_id",
    "path",
    "label",
    "generator_family",
    "release_date",
    "generator",
    "crop_path",
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


def crop_policy_id(size, align):
    """Identify a crop policy so caches for different policies cannot collide."""
    return f"crop{size}_align{align}"


def crop_dir(policy_id):
    return common.DATA_DIR / "preprocessed" / policy_id


def crop_path(file_id, policy_id):
    """Repo-root-relative path to a file_id's cached crop. Mirrors common.image_path's flattening."""
    return f"data/preprocessed/{policy_id}/{file_id.replace('/', '_')}.png"


def family_of(row):
    generator = (row["generator"] or "").strip()
    if row["label"] == common.REAL_LABEL:
        return AUTHENTIC_FAMILY
    return GENERATOR_FAMILIES[generator]


def build(policy_id, split=None, origin_dataset=None, label=None):
    """The sample as a panel manifest DataFrame, filtered as requested."""
    sample = common.load_sample()

    if split is not None:
        sample = sample[sample["split"] == split]
    if origin_dataset is not None:
        sample = sample[sample["origin_dataset"] == origin_dataset]
    if label is not None:
        sample = sample[sample["label"] == label]
    if sample.empty:
        raise SystemExit("no rows left after filtering -- check --split/--origin-dataset/--label")

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
    args = parser.parse_args()

    if not common.sample_exists():
        raise SystemExit("sample not imported; run imports/sample/authentic.py and fakes.py first")

    policy_id = crop_policy_id(args.crop_size, args.crop_align)
    manifest = build(policy_id, split=args.split, origin_dataset=args.origin_dataset, label=args.label)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(args.out, index=False)

    counts = manifest["generator_family"].value_counts()
    print(f"wrote {len(manifest)} rows -> {args.out}")
    print(f"crop policy: {policy_id}")
    print(f"labels: {dict(manifest['label'].value_counts())}")
    print("families: " + ", ".join(f"{fam}={n}" for fam, n in counts.items()))


if __name__ == "__main__":
    main()
