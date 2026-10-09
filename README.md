# anchor-date-forensics — `server-run` branch

Runs the detector panel over a manifest on the lab server — unattended, resumable, yielding to
other users — and serves a dashboard for checking the run's health.

This branch holds only the orchestration. It shares no history or files with `main`: the
detectors, their virtualenvs, the manifests and the experiments all live in a checkout of
`main`, which the runner drives from outside (`PANEL_REPO`, default `~/anchor-date-forensics`).
Each chunk is scored by `main`'s own `detectors/<name>/run_score.py` in a fresh subprocess, so a
score from the server is the same as a score from `main`'s `run_all.sh`.

- `runner/` — the pipeline: chunk queue (`state.py`), per-GPU workers (`worker.py`), the
  shared-machine governor (`governor.py`), chunk reassembly (`merge.py`), the dashboard
  (`server.py`, `static/`), and every tunable with its rationale (`config.py`).
- `runner/run.sh` — start / attach / status / stop / merge; see its header.

## Setup on the server

```bash
cd ~/anchor-date-forensics                     # clone of main, detector venvs installed
git worktree add ~/anchor-date-forensics-server-run server-run
cd ~/anchor-date-forensics-server-run
python3.12 -m venv .venv && .venv/bin/pip install -r requirements.txt
```

## Running

```bash
EXP=~/anchor-date-forensics/experiments/stratified-36k
./runner/run.sh smoke "$EXP/manifest.csv" "$EXP/scores"     # rehearse on 40 images
./runner/run.sh start "$EXP/manifest.csv" "$EXP/scores"     # the real run, detached
./runner/run.sh status
```

The dashboard binds to 127.0.0.1:8766; tunnel it with
`ssh -F ssh_config -L 8766:127.0.0.1:8766 remote` and open http://localhost:8766.
Job state and logs go to `var/` here; chunk scratch goes to `/data/aigenbench/panel/work`.
