"""Score a manifest with AEROBLADE (Ricker et al., CVPR 2024).

AEROBLADE is training-free: it measures how well an image survives a round trip through a
latent diffusion model's autoencoder. Images produced by such a model reconstruct with a
*smaller* error than authentic ones.

Scoring uses the repo's own distance code (aeroblade.distances.distance_from_config), and
the reconstruction step mirrors aeroblade.image.compute_reconstructions exactly. It is
reimplemented rather than called for three reasons, all of them fatal to an unattended run
on this machine:

  * aeroblade.misc.safe_mkdir prompts on stdin with input(), which would hang run_all.sh;
  * compute_reconstructions calls pipe.enable_model_cpu_offload(), which needs a CUDA
    accelerator;
  * it wraps the AE in torch.compile, which is unreliable on MPS.

Loading only the autoencoders also cuts the download from ~13 GB of full img2img pipelines
to ~0.6 GB, with identical numbers: same weights, same metric.

raw_score = max over the three autoencoders of aeroblade's returned distance. The repo's
LPIPS._postprocess already negates, so that maximum equals -min(raw LPIPS), i.e. the best
reconstruction across AEs, and higher means more synthetic -- the same orientation as the
rest of the panel.
"""

import argparse
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "aeroblade"
sys.path.insert(0, str(HERE.parent))       # detectors/panel_io.py

# aeroblade.distances builds its joblib cache with Memory(location="cache"), resolved
# against the CWD at import time. chdir first so the cache lands here rather than wherever
# run_all.sh happened to be invoked from.
os.chdir(HERE)

import panel_io  # noqa: E402
import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms.functional import to_pil_image, to_tensor  # noqa: E402

from diffusers import AutoencoderKL, VQModel  # noqa: E402
from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion_img2img import (  # noqa: E402
    retrieve_latents,
)

from aeroblade.data import ImageFolder  # noqa: E402  (upstream)
from aeroblade.distances import distance_from_config  # noqa: E402  (upstream)

UPSTREAM_REPO = "https://github.com/jonasricker/aeroblade"

# The three autoencoders from the paper. AutoencoderKL lives under vae/, Kandinsky's
# MoVQ under movq/.
AUTOENCODERS = [
    ("CompVis/stable-diffusion-v1-1", "vae", AutoencoderKL),
    ("stabilityai/stable-diffusion-2-base", "vae", AutoencoderKL),
    ("kandinsky-community/kandinsky-2-1", "movq", VQModel),
]
AUTH_HINT = (
    "   Accept the licence at https://huggingface.co/stabilityai/stable-diffusion-2-base\n"
    "   then: detectors/aeroblade/.venv/bin/huggingface-cli login\n"
    "   and re-run. Delete the stale scores first so the sidecar is rewritten."
)
DISTANCE_METRIC = "lpips_vgg_2"  # the second LPIPS layer, which the paper found best
SEED = 1


class GatedAutoencoder(Exception):
    """Raised when an autoencoder repo needs Hugging Face credentials we do not have."""

    def __init__(self, repo_id):
        super().__init__(repo_id)
        self.repo_id = repo_id


def is_auth_error(exc):
    """True for a 401/gated-repo failure, as opposed to a genuine load error.

    Stability gated every stabilityai/stable-diffusion-2* repo, so SD2's autoencoder now
    needs an accepted licence plus a token. That must not silently become a wrong score,
    and it must not block the other two autoencoders either.
    """
    text = f"{type(exc).__name__}: {exc}".lower()
    return any(
        marker in text
        for marker in ("401", "gated", "is not a valid model identifier", "authentication",
                       "restricted", "awaiting a review", "login")
    )


def slug(repo_id):
    return repo_id.replace("/", "__")


@torch.no_grad()
def reconstruct(paths, repo_id, subfolder, cls, out_dir, device, batch_size):
    """AE round-trip, mirroring aeroblade.image.compute_reconstructions.

    fp32 rather than upstream's fp16: there is no CUDA here, and fp16 on CPU is
    unusably slow. Recorded as a deviation in the sidecar.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    targets = [out_dir / f"{p.stem}.png" for p in paths]
    todo = [(src, dst) for src, dst in zip(paths, targets) if not dst.exists()]

    if todo:
        try:
            # No use_safetensors=True: the 2022-era CompVis repo ships only .bin, and
            # diffusers already prefers safetensors wherever they exist.
            ae = cls.from_pretrained(repo_id, subfolder=subfolder)
        except Exception as exc:
            if is_auth_error(exc):
                raise GatedAutoencoder(repo_id) from exc
            raise SystemExit(f"could not load {repo_id} ({subfolder}): {exc}") from exc
        ae = ae.to(device).eval()
        decode_dtype = next(iter(ae.post_quant_conv.parameters())).dtype
        generator = torch.Generator().manual_seed(SEED)

        for start in range(0, len(todo), batch_size):
            chunk = todo[start:start + batch_size]
            batch = torch.stack([to_tensor(Image.open(src).convert("RGB")) for src, _ in chunk])
            batch = batch.to(device, dtype=ae.dtype) * 2.0 - 1.0

            latents = retrieve_latents(ae.encode(batch), generator=generator)
            if isinstance(ae, VQModel):
                recons = ae.decode(latents.to(decode_dtype), force_not_quantize=True, return_dict=False)[0]
            else:
                recons = ae.decode(latents.to(decode_dtype), return_dict=False)[0]
            recons = (recons / 2 + 0.5).clamp(0, 1)

            for recon, (_, dst) in zip(recons.cpu(), chunk):
                to_pil_image(recon).save(dst)

        del ae
        if device == "mps":
            torch.mps.empty_cache()

    return targets


def main():
    parser = panel_io.add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--recon-dir", type=Path, default=HERE / "reconstructions")
    parser.add_argument("--require-all-aes", action="store_true",
                        help="fail instead of degrading when an autoencoder needs HF credentials")
    args = parser.parse_args()
    # The VAEs are the memory-heavy step; 16 images of 200x200 at once is plenty.
    batch_size = min(args.batch_size, 8)

    manifest = panel_io.load_manifest(args.manifest, args.limit)
    originals = list(manifest["image_file"])
    policy = originals[0].parent.name

    distance = distance_from_config(DISTANCE_METRIC, batch_size=batch_size, num_workers=0)
    ds_original = ImageFolder(paths=originals)

    per_ae = {}
    gated = []
    with panel_io.Timer() as timer:
        for repo_id, subfolder, cls in AUTOENCODERS:
            print(f"[{repo_id}] reconstructing...", flush=True)
            try:
                recons = reconstruct(
                    originals, repo_id, subfolder, cls,
                    args.recon_dir / policy / slug(repo_id), args.device, batch_size,
                )
            except GatedAutoencoder as exc:
                gated.append(exc.repo_id)
                print(f"[{repo_id}] SKIPPED -- needs Hugging Face credentials", flush=True)
                continue
            result, files = distance.compute(ds_a=ds_original, ds_b=ImageFolder(paths=recons))
            per_ae[repo_id] = result[DISTANCE_METRIC].flatten().to(torch.float32)
            assert [Path(f).stem for f in files] == [p.stem for p in originals], "file order drifted"

    if not per_ae:
        raise SystemExit("no autoencoder could be loaded; nothing to score")
    if gated and args.require_all_aes:
        raise SystemExit(
            f"these autoencoders need Hugging Face credentials: {', '.join(gated)}\n" + AUTH_HINT
        )

    # repo_id="max" in upstream's compute_distances: the best reconstruction across AEs.
    scores = torch.stack(list(per_ae.values())).max(dim=0).values.tolist()

    panel_io.write_scores(args.out, manifest["image_id"].tolist(), scores)
    panel_io.write_meta(
        args.out,
        detector="aeroblade",
        upstream_repo=UPSTREAM_REPO,
        upstream_dir=UPSTREAM,
        weights=[],
        score_semantics=f"neg_{DISTANCE_METRIC}_max_over_aes",
        manifest_path=args.manifest,
        device=args.device,
        n_rows=len(manifest),
        elapsed_s=timer.elapsed,
        extra={
            "autoencoders_configured": [r for r, _, _ in AUTOENCODERS],
            "autoencoders_used": list(per_ae),
            "autoencoders_skipped_needs_hf_auth": gated,
            "distance_metric": DISTANCE_METRIC,
            "seed": SEED,
            "deviations_from_upstream": [
                "fp32 instead of fp16 (no CUDA available; fp16 on CPU is unusably slow)",
                "autoencoders loaded directly instead of via AutoPipelineForImage2Image",
                "no torch.compile, no enable_model_cpu_offload",
            ],
            "preprocessing": "shared 200x200 crop; ToImage + float32 scale (aeroblade ImageFolder default)",
        },
    )
    panel_io.report(args.out, scores, timer.elapsed)
    if gated:
        print(
            f"\n!! DEGRADED: scored with {len(per_ae)}/{len(AUTOENCODERS)} autoencoders.\n"
            f"   unavailable: {', '.join(gated)}\n" + AUTH_HINT,
            file=sys.stderr,
        )


if __name__ == "__main__":
    main()
