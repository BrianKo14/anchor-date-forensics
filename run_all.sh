#!/usr/bin/env bash
#
# Score every image in manifest.csv with all four detectors.
#
#   ./run_all.sh                    # cpu (default), ~57 min for 288 images
#   DEVICE=mps ./run_all.sh         # faster, but not bit-reproducible against cpu scores
#   REBUILD_MANIFEST=1 ./run_all.sh # regenerate manifest.csv from the sample
#
# Each detector lives in its own virtualenv because their dependency sets are mutually
# incompatible. Their interpreters are invoked by absolute path -- there is no activate
# or deactivate anywhere, so nothing leaks between environments or into the caller's shell.
#
# Writes scores/<name>.csv (image_id,raw_score) plus a .meta.json sidecar recording the
# weights, commit and pins each score came from. merge_scores.ipynb joins them.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEVICE="${DEVICE:-cpu}"
MANIFEST="$ROOT/manifest.csv"
SCORES="$ROOT/scores"
mkdir -p "$SCORES"

echo "device: $DEVICE"
echo

# --- inputs (both idempotent, so re-runs are cheap) -----------------------------------
if [ ! -f "$MANIFEST" ] || [ -n "${REBUILD_MANIFEST:-}" ]; then
  echo "--- building manifest"
  "$ROOT/.venv/bin/python" "$ROOT/imports/build_manifest.py" --out "$MANIFEST"
else
  echo "--- manifest: $(($(wc -l < "$MANIFEST") - 1)) rows"
fi

echo "--- crop cache"
"$ROOT/.venv/bin/python" "$ROOT/imports/preprocess_crop_cache.py" --manifest "$MANIFEST"
echo

# --- the detectors --------------------------------------------------------------------
# DMimageDetection ships two independently-trained checkpoints; both are scored, because
# progan is the stronger GAN detector and latent the stronger diffusion detector.
# name : detector-dir : extra args
PANEL=(
  "cnndetection:cnndetection:"
  "univfd:univfd:"
  "dmimagedetection_progan:dmimagedetection:--model Grag2021_progan"
  "dmimagedetection_latent:dmimagedetection:--model Grag2021_latent"
  "aeroblade:aeroblade:"
)

started=$SECONDS
for entry in "${PANEL[@]}"; do
  name="${entry%%:*}"; rest="${entry#*:}"
  dir="${rest%%:*}"; extra="${rest#*:}"

  echo "=== $name ==="
  t0=$SECONDS
  # shellcheck disable=SC2086  # $extra is an intentional argument list
  "$ROOT/detectors/$dir/.venv/bin/python" "$ROOT/detectors/$dir/run_score.py" \
      --manifest "$MANIFEST" \
      --out "$SCORES/$name.csv" \
      --device "$DEVICE" $extra
  echo "    [${name}: $((SECONDS - t0))s]"
  echo
done

echo "done in $((SECONDS - started))s -- now run merge_scores.ipynb"
for f in "$SCORES"/*.csv; do
  printf '  %-34s %s rows\n' "${f#$ROOT/}" "$(($(wc -l < "$f") - 1))"
done
