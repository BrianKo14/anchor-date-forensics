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

A miniature of AI-GenBench: **72 authentic + 72 fake** images, exactly 50/50, with 2 images from
each of the 36 benchmark generators. It lands in `data/` (gitignored).

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
- **Small and narrow**: 72 + 72, `validation` split only, drawn from one of 15 shards.
- **Not byte-reproducible**: LAION link rot means a rerun picks different images.
- **No ImageNet**, though the AI-GenBench README lists it — its shipped file-id lists contain none.
- **RAISE needs `RAISE_urls.csv`** fetched by hand from
  [loki.disi.unitn.it/RAISE](http://loki.disi.unitn.it/RAISE/confirm.php?package=all); the step is
  skipped if the file is absent.
- **No archival scans.** RAISE is camera-raw, not digitized print or film.

## Layout

```
imports/sample/     one-time importers
  common.py        paths, constants, manifest I/O
  aigenbench.py    the benchmark repo: file-id lists, LAION filelist, generator registry
  imaging.py       decoding and JPEG normalization -- one prepare_image, used by both halves
  authentic.py     entry point for the real half
  fakes.py         entry point for the synthetic half
sample.ipynb       read the sample
data/              the sample itself (gitignored)
```

## Notes

- `HF_HOME` points at `.cache/huggingface/` inside the repo so dataset downloads stay local. This can
  grow large; it is gitignored.
- `AI-GenBench/` is cloned on first run (shallow) and gitignored, along with `.venv/`, `.cache/` and
  `data/`.
- Notebook outputs are stripped from commits by the nbstripout git filter (see Setup). If
  you skipped that step, `git diff` on the `.ipynb` will show a huge blob — run the install
  line and re-stage.
