"""Score a manifest with DMimageDetection (Corvi et al., ICASSP 2023).

Calls the repo's own loader (get_method_here.def_model) and its own normalization
(normalization.get_list_norm), and reduces the output exactly as test_code/main.py does.
main.py itself is not imported: it calls main() at module scope and walks a directory
tree of its own.

The repo ships two independently-trained checkpoints, and both are panel members:
  Grag2021_progan  GAN-trained
  Grag2021_latent  latent-diffusion-trained
Select with --model; run_all.sh invokes this script once per checkpoint so that the
--manifest/--out contract stays identical across all four detectors.

raw_score = the logit, positive meaning synthetic (per the upstream README). Because
res50stride1 does not downsample in its stem, a 200x200 input yields a spatial logit map
rather than a scalar; main.py's mean over that map is reproduced here.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "DMimageDetection"
TEST_CODE = UPSTREAM / "test_code"
sys.path.insert(0, str(HERE.parent))       # detectors/panel_io.py
sys.path.insert(0, str(TEST_CODE))         # the repo's flat modules

import panel_io  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image  # noqa: E402

# Upstream bug, worked around without patching the clone: get_method_here.def_model does
# `import networks.networks.resnet_mod`, but the file is at test_code/networks/resnet_mod.py
# -- there is no nested networks/networks/. Aliasing the package makes the repo's own
# loader importable as written.
import networks  # noqa: E402  (upstream)
sys.modules.setdefault("networks.networks", networks)

from get_method_here import get_method_here, def_model  # noqa: E402  (upstream)
from normalization import CenterCropNoPad, get_list_norm  # noqa: E402  (upstream)

UPSTREAM_REPO = "https://github.com/grip-unina/DMimageDetection"
MODELS = ["Grag2021_progan", "Grag2021_latent"]
DEFAULT_WEIGHTS_DIR = TEST_CODE / "weights"


def build(model_name, weights_dir, device):
    _, model_path, arch, norm_type, patch_size = get_method_here(model_name, weights_path=str(weights_dir))
    model = def_model(arch, model_path, localize=False).to(device).eval()

    # Mirrors main.py's transform assembly. patch_size is None for both checkpoints, so no
    # crop is applied -- correct here, since the input is already the shared 200x200 crop.
    steps = []
    if patch_size is not None:
        steps.append(CenterCropNoPad(patch_size))
    transform = transforms.Compose(steps + get_list_norm(norm_type))
    return model, transform, Path(model_path)


def reduce_logits(out):
    """main.py's reduction: pick the synthetic channel, then average any spatial map."""
    if out.shape[1] == 1:
        out = out[:, 0]
    elif out.shape[1] == 2:
        out = out[:, 1] - out[:, 0]
    else:
        raise SystemExit(f"unexpected model output shape {out.shape}")
    return np.mean(out, (1, 2)) if out.ndim > 1 else out


def main():
    parser = panel_io.add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--model", choices=MODELS, default="Grag2021_latent")
    parser.add_argument("--weights-dir", type=Path, default=DEFAULT_WEIGHTS_DIR)
    args = parser.parse_args()

    if not (args.weights_dir / args.model / "model_epoch_best.pth").exists():
        raise SystemExit(
            f"weights not found under {args.weights_dir}\n"
            "run detectors/dmimagedetection/fetch_weights.sh"
        )

    manifest = panel_io.load_manifest(args.manifest, args.limit)
    model, transform, weights_path = build(args.model, args.weights_dir, args.device)

    scores = []
    with panel_io.Timer() as timer, torch.no_grad():
        for start in range(0, len(manifest), args.batch_size):
            paths = manifest["image_file"].iloc[start:start + args.batch_size]
            batch = torch.stack([transform(Image.open(p).convert("RGB")) for p in paths])
            out = model(batch.to(args.device)).cpu().numpy()
            scores.extend(reduce_logits(out).tolist())

    panel_io.write_scores(args.out, manifest["image_id"].tolist(), scores)
    panel_io.write_meta(
        args.out,
        detector=f"dmimagedetection_{args.model.replace('Grag2021_', '')}",
        upstream_repo=UPSTREAM_REPO,
        upstream_dir=UPSTREAM,
        weights=[weights_path],
        score_semantics="logit_spatial_mean",
        manifest_path=args.manifest,
        device=args.device,
        n_rows=len(manifest),
        elapsed_s=timer.elapsed,
        extra={
            "checkpoint": args.model,
            "preprocessing": "shared 200x200 crop; get_list_norm('resnet'); no additional crop",
        },
    )
    panel_io.report(args.out, scores, timer.elapsed)


if __name__ == "__main__":
    main()
