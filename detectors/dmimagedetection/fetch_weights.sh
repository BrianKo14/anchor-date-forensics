#!/usr/bin/env bash
# Fetch DMimageDetection pretrained weights (Grag2021_progan, Grag2021_latent).
#
# The checkpoints live on Google Drive behind the virus-scan interstitial, so plain curl
# gets HTML rather than the archive; gdown handles the confirm token. If this fails, the
# fallback is a manual browser download -- see the message at the bottom.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/DMimageDetection/test_code/weights"
FILE_ID="1sAoAuOGCWS4dAMBhDkRHgBf4SgBgvkVf"
ZIP="$HERE/weights.zip"

if [ -f "$DEST/Grag2021_progan/model_epoch_best.pth" ] && [ -f "$DEST/Grag2021_latent/model_epoch_best.pth" ]; then
  echo "skip (weights already present): $DEST"
  exit 0
fi

mkdir -p "$DEST"
if [ ! -s "$ZIP" ]; then
  "$HERE/.venv/bin/python" -m gdown "$FILE_ID" -O "$ZIP" || {
    echo
    echo "gdown failed. Download manually from:"
    echo "  https://drive.google.com/file/d/$FILE_ID/view"
    echo "save it as $ZIP and re-run this script."
    exit 1
  }
fi

unzip -o -q "$ZIP" -d "$DEST"
# The archive nests everything under weights/; flatten if so.
if [ -d "$DEST/weights" ]; then
  mv "$DEST"/weights/* "$DEST"/ && rmdir "$DEST/weights"
fi
echo "--- weights ---"; find "$DEST" -name "*.pth" -exec shasum -a 256 {} \;
