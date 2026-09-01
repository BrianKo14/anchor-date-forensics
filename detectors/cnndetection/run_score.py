"""Score a manifest with CNNDetection (Wang et al., CVPR 2020).

Calls the repo's own model code (networks.resnet.resnet50) and replicates demo.py's
model load and transform; demo.py itself is not imported, since it is a CLI script that
parses argv and scores a single file at import time.

raw_score = the pre-sigmoid logit. Higher means more synthetic. The logit rather than
.sigmoid() because the probability saturates at 1.0 in float32 and throws away the
ordering PAV needs downstream; sigmoid(raw_score) recovers demo.py's number exactly.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE / "CNNDetection"
sys.path.insert(0, str(HERE.parent))       # detectors/panel_io.py
sys.path.insert(0, str(UPSTREAM))          # the repo's own networks/ package

import panel_io  # noqa: E402
import torch  # noqa: E402
import torchvision.transforms as transforms  # noqa: E402
from PIL import Image  # noqa: E402

from networks.resnet import resnet50  # noqa: E402  (upstream)

UPSTREAM_REPO = "https://github.com/PeterWang512/CNNDetection"
DEFAULT_WEIGHTS = UPSTREAM / "weights" / "blur_jpg_prob0.5.pth"

# demo.py's transform. No crop or resize here: the shared cache has already produced a
# 200x200 crop, and this detector keys on resampling artifacts, so adding a resize would
# manufacture exactly the signal it is looking for.
TRANSFORM = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def load_model(weights, device):
    model = resnet50(num_classes=1)
    state_dict = torch.load(weights, map_location="cpu")
    model.load_state_dict(state_dict["model"])
    model.eval()
    return model.to(device)


def main():
    parser = panel_io.add_common_args(argparse.ArgumentParser(description=__doc__))
    parser.add_argument("--weights", type=Path, default=DEFAULT_WEIGHTS)
    args = parser.parse_args()

    if not args.weights.exists():
        raise SystemExit(f"weights not found: {args.weights}\nrun detectors/cnndetection/fetch_weights.sh")

    manifest = panel_io.load_manifest(args.manifest, args.limit)
    model = load_model(args.weights, args.device)

    scores = []
    with panel_io.Timer() as timer, torch.no_grad():
        for start in range(0, len(manifest), args.batch_size):
            paths = manifest["image_file"].iloc[start:start + args.batch_size]
            batch = torch.stack([TRANSFORM(Image.open(p).convert("RGB")) for p in paths])
            logits = model(batch.to(args.device)).flatten()
            scores.extend(logits.cpu().tolist())

    panel_io.write_scores(args.out, manifest["image_id"].tolist(), scores)
    panel_io.write_meta(
        args.out,
        detector="cnndetection",
        upstream_repo=UPSTREAM_REPO,
        upstream_dir=UPSTREAM,
        weights=[args.weights],
        score_semantics="pre_sigmoid_logit",
        manifest_path=args.manifest,
        device=args.device,
        n_rows=len(manifest),
        elapsed_s=timer.elapsed,
        extra={"preprocessing": "shared 200x200 crop; ToTensor + imagenet Normalize (demo.py)"},
    )
    panel_io.report(args.out, scores, timer.elapsed)


if __name__ == "__main__":
    main()
