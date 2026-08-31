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

### Running the notebook

Open `ai-genbench-exploration.ipynb` **with the repo root as the working directory**:

```bash
.venv/bin/jupyter lab ai-genbench-exploration.ipynb
```

## Notes

- points `HF_HOME` at `.cache/huggingface/` inside the repo so dataset downloads stay
  local. This can grow large; it is gitignored.

- The dataset is loaded with `streaming=True` and only a few-image sample is pulled, so a full run needs network but not the full ~180K-image download.

- `AI-GenBench/`, `.venv/`, and `.cache/` are gitignored.

- Notebook outputs are stripped from commits by the nbstripout git filter (see Setup). If
  you skipped that step, `git diff` on the `.ipynb` will show a huge blob — run the install
  line and re-stage.
