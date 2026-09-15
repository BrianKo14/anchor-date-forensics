# anchor-date-forensics — `downloader` branch

Full-dataset (360,000 image) build for the AI-GenBench benchmark, shelved for now in favor of the
612+612 in-repo sample that `main` runs experiments against. Picks back up whenever the prototype
needs the whole dataset rather than the sample.

This branch is deliberately scoped to just what the downloader needs — it does not share history
or files with `main`'s experiment code (`lr/`, `detectors/`, notebooks, etc.).

- `downloader/` — the pipeline itself. `common.py`, `aigenbench.py` and `imaging.py` are copied in
  flat from `main`'s `imports/sample/` so the full dataset stays byte-compatible with the sample;
  see the note at the top of `downloader/config.py`.
- `RAISE_urls.csv` — RAISE's pinned URL list, needed by `downloader/sources/raise_tiff.py`.
- `downloader/data_readme.md` — what gets written to disk and how the run resumes.
- `downloader/home_readme.md` — for anyone on the shared server who stumbles onto the job.

Run with `./downloader/run.sh start` (see that file's header for `attach`/`status`/`stop`/`smoke`).
Needs a venv at `.venv/` with `requirements.txt` installed.
