# GAMI: Time-calibrated deepfake detection

Prototyping for a time-calibrated AI-generation detection system: given a cryptographic
anchor date **T**, estimate the likelihood ratio that an image came from some image
generator available by T.

## Setup

Python 3.12.

```bash
python -m venv .venv
.venv/bin/pip install -r requirements.txt
.venv/bin/python -m ipykernel install --user --name anchor-date-forensics
.venv/bin/nbstripout --install --attributes .gitattributes   # strip notebook outputs on commit
```

Note on `nbstripout`: the last line wires up a git filter: your working copy keeps its cell outputs, but
committed versions have them removed.

## The sample

A miniature of AI-GenBench: **144 authentic + 144 fake** images, exactly 50/50, with 4 images from
each of the 36 benchmark generators. It lands in `data/` (gitignored).

It is partitioned into a **train** and a **val** half, 72 + 72 each way, recorded per row in the
manifests' `split` column. The partition is stratified — every generator contributes 2 images to
each side, and the three authentic sources keep their proportions — because a per-family score model
cannot be calibrated on a family that landed entirely on one side. Both halves go through the same
`common.assign_splits`, seeded per stratum so that growing the sample leaves existing assignments
untouched.

Note that `split` (train/val) and the AI-GenBench split the ids are drawn from (`validation`, in the
manifest filenames) are different axes; in code the latter is `common.BENCHMARK_SPLIT`.

Importing:

```bash
.venv/bin/python imports/sample/authentic.py   # COCO + LAION + RAISE -> data/authentic
.venv/bin/python imports/sample/fakes.py       # AI-GenBench fake part -> data/fakes
```

Both are re-runnable and skip whatever is already on disk, so a second run costs nothing. The
authentic half is fetched image by image (no origin archives); the fake half reads the benchmark's
parquet directly, downloading only the row groups it draws from (~51 MB, against a 7 GB split).

Then explore it with `sample.ipynb`, with the repo root as the working directory:

```bash
.venv/bin/jupyter lab sample.ipynb
```

### Caveats

- **Not content-paired.** AI-GenBench prioritizes reals sharing a caption, mask, or inpainting source
  with a specific fake (its `paired_real_images` column). We sample the two halves independently, so
  real-vs-fake comparisons are still confounded by image content.
- **Normalized to JPEG q95**, i.e. the benchmark's `make_jpeg_dataset = True` variant. The shipped
  default is `False`, which leaves fakes as native PNG/JPEG/WEBP against near-all-JPEG reals — a
  format difference that separates the classes on its own.
- **Small and narrow**: 144 + 144, `validation` split only, drawn from one of 15 shards.
- **Not byte-reproducible**: LAION link rot means a rerun picks different images. The importer
  oversamples a shuffled filelist to reach its 68 LAION images, so how many URLs it walks varies.
- **No ImageNet**, though the AI-GenBench README lists it — its shipped file-id lists contain none.
- **RAISE needs `RAISE_urls.csv`** fetched by hand from
  [loki.disi.unitn.it/RAISE](http://loki.disi.unitn.it/RAISE/confirm.php?package=all); the step is
  skipped if the file is absent.
- **No archival scans.** RAISE is camera-raw, not digitized print or film.
- **The detector panel uses 200x200 crops, not native resolution.** All detectors read the shared
  `crop200_align16` cache. This avoids padding artifacts from `CenterCrop(224)` on small images,
  avoids huge full-resolution activations, and keeps panel scores comparable. Crop origins are
  aligned to 16, outputs are lossless PNG, and CLIP ViT-L/14 is uniformly upscaled 200->224.

## Detector panel

Four published detectors -- CNNDetection, UniversalFakeDetect, DMimageDetection (two
checkpoints) and AEROBLADE -- each in its own virtualenv, producing raw pre-threshold scores.

```sh
./run_all.sh                # score every image in manifest.csv (~57 min on cpu)
DEVICE=mps ./run_all.sh     # faster; cpu is the default and is bit-reproducible
```

This writes `scores/<detector>.csv` (`image_id,raw_score`) plus a `.meta.json` sidecar per
detector recording the weights, upstream commit and library pins the scores came from.
`merge_scores.ipynb` joins them into `scores/master_scores.csv` and plots the results.

Scores are logits (or, for AEROBLADE, a negated reconstruction distance), never binarized
verdicts, and all are oriented **higher = more synthetic**.

Setup, weight provenance, pin rationale and the upstream bugs worked around are in
[`detectors/README.md`](detectors/README.md). One caveat carried by the current scores:
AEROBLADE ran with 2 of its 3 autoencoders, because Stability gated the
`stabilityai/stable-diffusion-2*` repos.

## Layout

```
imports/sample/     one-time importers
  common.py        paths, constants, manifest I/O, train/val assignment
  aigenbench.py    the benchmark repo: file-id lists, LAION filelist, generator registry
  imaging.py       decoding and JPEG normalization -- one prepare_image, used by both halves
  authentic.py     entry point for the real half
  fakes.py         entry point for the synthetic half
build_manifest.py          sample -> manifest.csv (+ the generator family map)
preprocess_crop_cache.py   the crop cache the panel reads
sample.ipynb       read the sample
data/              the sample itself (gitignored)
detectors/         the detector panel: one venv, clone and run_score.py each
manifest.csv       image_id, path, label, generator_family, release_date, generator, crop_path
run_all.sh         score manifest.csv with all four detectors
scores/            one <detector>.csv + .meta.json each, and master_scores.csv
merge_scores.ipynb join the scores and look at them
```

`build_manifest.py` and `preprocess_crop_cache.py` sit in `imports/`, not `imports/sample/`:
`imports/sample/` is the one-time construction of the sample itself, while those two are
re-run whenever the manifest or crop policy changes.

## Notes

- `HF_HOME` points at `.cache/huggingface/` inside the repo so dataset downloads stay local. This can
  grow large; it is gitignored.
- `AI-GenBench/` is cloned on first run (shallow) and gitignored, along with `.venv/`, `.cache/` and
  `data/`.
- Notebook outputs are stripped from commits by the nbstripout git filter (see Setup). If
  you skipped that step, `git diff` on the `.ipynb` will show a huge blob — run the install
  line and re-stage.
