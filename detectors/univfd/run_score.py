"""Score a manifest with UniversalFakeDetect / UnivFD (Ojha et al., CVPR 2023).

Calls the repo's own model factory (models.get_model) and loads the linear probe the
repo ships. validate.py is deliberately not imported: it is a CLI eval script, and it
imports scipy.ndimage.filters, which scipy removed in 1.15.

raw_score = the fc logit, i.e. CLIPModel.forward's output before .sigmoid(). Higher
means more synthetic.

Preprocessing note: this is the one detector that cannot read the shared 200x200 crop
directly -- CLIP ViT-L/14 has fixed positional embeddings for a 224x224 input. Every
image is therefore bicubic-upscaled 200 -> 224 *uniformly*, so the resampling cannot
correlate with class. Upstream's CenterCrop(224) would instead zero-pad the 66 sub-224
images in this sample, 66 of which are fakes, turning padding into a class cue.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "UniversalFakeDetect"
sys.path.insert(0, str(HERE.parent))       # detectors/panel_io.py
sys.path.insert(0, str(UPSTREAM))          # the repo's own models/ package

import panel_io  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image  # noqa: E402

from models import get_model  # noqa: E402  (upstream)

UPSTREAM_REPO = "https://github.com/WisconsinAIVision/UniversalFakeDetect"
ARCH = "CLIP:ViT-L/14"
DEFAULT_WEIGHTS = UPSTREAM / "pretrained_weights" / "fc_weights.pth"
# Where the vendored clip.py caches the OpenAI backbone (models/clip/clip.py::_download).
CLIP_BACKBONE = Path.home() / ".cache" / "clip" / "ViT-L-14.pt"

# MEAN["clip"] / STD["clip"] from upstream validate.py.
TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.48145466, 0.4578275, 0.40821073),
                         std=(0.26862954, 0.26130258, 0.27577711)),
])


def load_model(weights, device):
    model = get_model(ARCH)
    model.fc.load_state_dict(torch.load(weights, map_location="cpu"))
    model.eval()
    return model.to(device)


def main():
    parser = panel_io.add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}\n(fc_weights.pth ships inside the repo)")

    manifest = panel_io.load_manifest(args.manifest, args.limit)
    model = load_model(args.weights, args.device)

    scores = []
    with panel_io.Timer() as timer, torch.no_grad():
        for start in range(0, len(manifest), args.batch_size):
            paths = manifest["image_file"].iloc[start:start + args.batch_size]
            batch = torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p in paths])
            logits = model(batch.to(args.device)).flatten()
            scores.extend(logits.cpu().tolist())

    weights = [args.weights] + ([CLIP_BACKBONE] if CLIP_BACKBONE.exists() else [])
    panel_io.write_scores(args.out, manifest["image_id"].tolist(), scores)
    panel_io.write_meta(
        args.out,
        detector="univfd",
        upstream_repo=UPSTREAM_REPO,
        upstream_dir=UPSTREAM,
        weights=weights,
        score_semantics="fc_logit",
        manifest_path=args.manifest,
        device=args.device,
        n_rows=len(manifest),
        elapsed_s=timer.elapsed,
        extra={
            "arch": ARCH,
            "preprocessing": "shared 200x200 crop; uniform bicubic resize 200->224; CLIP Normalize",
        },
    )
    panel_io.report(args.out, scores, timer.elapsed)


if __name__ == "__main__":
    main()
