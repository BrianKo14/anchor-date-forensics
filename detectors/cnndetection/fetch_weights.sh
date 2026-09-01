#!/usr/bin/env bash
# Fetch CNNDetection pretrained weights.
#
# Upstream weights/download_weights.sh uses wget, which is not installed on macOS
# by default; these are the same two Dropbox objects with dl=1 so curl gets the
# file rather than the HTML interstitial.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEST="$HERE/CNNDetection/weights"
mkdir -p "$DEST"
fetch() { # url dest
  if [ -s "$2" ]; then echo "skip (exists): $2"; return; fi
  echo "fetching $2"
  curl -fL --retry 3 --progress-bar -o "$2.part" "$1"
  mv "$2.part" "$2"
}
fetch "https://www.dropbox.com/s/2g2jagq2jn1fd0i/blur_jpg_prob0.5.pth?dl=1" "$DEST/blur_jpg_prob0.5.pth"
fetch "https://www.dropbox.com/s/h7tkpcgiwuftb6g/blur_jpg_prob0.1.pth?dl=1" "$DEST/blur_jpg_prob0.1.pth"
echo "--- sha256 ---"; shasum -a 256 "$DEST"/*.pth
