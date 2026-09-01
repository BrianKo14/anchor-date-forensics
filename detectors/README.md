# Detector panel

Four published AI-image detectors, frozen, producing raw pre-threshold scores.

The panel is **shared infrastructure, installed once**. Every experiment reuses these same
virtualenvs, clones and weights: reinstalling per experiment would cost ~13 GB each time and,
worse, would break the guarantee that makes scores comparable across experiments. An
experiment is a manifest plus an output directory — never new detector code.

| detector | paper | score | orientation |
|---|---|---|---|
| `cnndetection` | Wang et al., CVPR 2020 | pre-sigmoid logit | higher = more synthetic |
| `univfd` | Ojha et al., CVPR 2023 | fc logit | higher = more synthetic |
| `dmimagedetection_progan` | Corvi et al., ICASSP 2023 | logit, spatial mean | higher = more synthetic |
| `dmimagedetection_latent` | Corvi et al., ICASSP 2023 | logit, spatial mean | higher = more synthetic |
| `aeroblade` | Ricker et al., CVPR 2024 | −LPIPS(vgg,layer 2), max over AEs | higher = more synthetic |

DMimageDetection contributes two panel members: it ships two independently-trained
checkpoints, one on GAN output and one on latent-diffusion output, and their disagreement is
itself a family-attribution signal.

Scores are **logits, not probabilities**. A sigmoid saturates at 1.0 in float32 and discards
the ordering in the upper tail, which is precisely the region the likelihood ratio depends on.
`sigmoid(raw_score)` recovers each repo's published number.

## Setup

Requires Python 3.12 (`/opt/homebrew/opt/python@3.12/bin/python3.12`) and ~13 GB of disk.

```sh
# 1. clone the four upstreams
git clone --depth 1 https://github.com/PeterWang512/CNNDetection.git        detectors/cnndetection/CNNDetection
git clone --depth 1 https://github.com/WisconsinAIVision/UniversalFakeDetect.git detectors/univfd/UniversalFakeDetect
git clone --depth 1 https://github.com/grip-unina/DMimageDetection.git      detectors/dmimagedetection/DMimageDetection
git clone --depth 1 https://github.com/jonasricker/aeroblade.git            detectors/aeroblade/aeroblade

# 2. one isolated env each
for d in cnndetection univfd dmimagedetection aeroblade; do
  /opt/homebrew/opt/python@3.12/bin/python3.12 -m venv "detectors/$d/.venv"
  "detectors/$d/.venv/bin/python" -m pip install -r "detectors/$d/requirements.txt"
done
detectors/aeroblade/.venv/bin/python -m pip install -e detectors/aeroblade/aeroblade --no-deps

# 3. weights
bash detectors/cnndetection/fetch_weights.sh        # 2 x 282 MB from Dropbox
bash detectors/dmimagedetection/fetch_weights.sh    # via gdown, Google Drive
# univfd needs nothing: fc_weights.pth ships in the repo. The CLIP ViT-L/14 backbone
# (~890 MB) downloads to ~/.cache/clip on first run.
# aeroblade needs nothing: it is training-free. The autoencoders (~0.6 GB) come from
# Hugging Face on first run.

# 4. record what got installed
.venv/bin/python detectors/build_panel_json.py
```

## Environment notes

These are not arbitrary pins.

- **`torch==2.5.1` everywhere.** torch 2.6 flipped `torch.load` to `weights_only=True`, which
  breaks the unguarded `torch.load(ckpt)` in CNNDetection and DMimageDetection. A single pin
  across the panel is also what "frozen" means here.
- **`setuptools==80.9.0` in univfd.** The vendored `models/clip/clip.py` does
  `from pkg_resources import packaging`, and setuptools 81 removed `pkg_resources`. Pinning
  keeps the upstream clone unpatched.
- **aeroblade does not use upstream's `requirements.txt`.** That file is a CUDA pip-freeze
  (`nvidia-*`, `triton`, `torch==2.1.2`, which has no cp312 macOS arm64 wheel) and cannot
  install here. `detectors/aeroblade/requirements.txt` is the curated equivalent. `pyiqa` is
  in it only because `aeroblade/distances.py` imports it at module scope.

## Known upstream issues worked around

None of the clones are patched; the workarounds live in our `run_score.py` files.

- **DMimageDetection** — `get_method_here.def_model` does `import networks.networks.resnet_mod`,
  but the file is at `test_code/networks/resnet_mod.py`; there is no nested `networks/networks/`.
  `run_score.py` aliases the package in `sys.modules` so the repo's own loader works as written.
- **AEROBLADE** — `misc.safe_mkdir` prompts on stdin with `input()`, which would hang an
  unattended run; `compute_reconstructions` also calls `enable_model_cpu_offload()` (needs CUDA)
  and wraps the AE in `torch.compile` (unreliable on MPS). `run_score.py` therefore loads the
  autoencoders directly and mirrors the reconstruction maths, while still scoring with the
  repo's own `distance_from_config`. This also cuts the download from ~13 GB to ~0.6 GB.
- **UniversalFakeDetect** — `validate.py` imports `scipy.ndimage.filters`, removed in scipy
  1.15. Only `models/` is imported, never `validate.py`.

## AEROBLADE and Hugging Face credentials

Stability has gated every `stabilityai/stable-diffusion-2*` repo, so one of AEROBLADE's three
autoencoders needs an accepted licence and a token. Without one, `run_score.py` **degrades
loudly**: it scores with the two reachable autoencoders, prints a warning, and records
`autoencoders_used` / `autoencoders_skipped_needs_hf_auth` in the sidecar. To restore the
third:

```sh
# accept the licence at https://huggingface.co/stabilityai/stable-diffusion-2-base first
detectors/aeroblade/.venv/bin/huggingface-cli login
rm scores/aeroblade.*      # so the sidecar is rewritten
./run_all.sh
```

Pass `--require-all-aes` to make a missing autoencoder fatal instead.

## Adding a detector

Create `detectors/<name>/` with a `requirements.txt`, its own `.venv`, the upstream clone, and
a `run_score.py` that takes `--manifest` / `--out` / `--device`, writes `image_id,raw_score`,
and calls `panel_io.write_meta`. Add it to `PANEL` in `build_panel_json.py` and to `PANEL` in
`run_all.sh`. Nothing else needs to change.
