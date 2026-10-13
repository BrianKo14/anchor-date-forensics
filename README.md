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
- `runner/selfdesc.py` — a second, CPU job on the same machinery: forensic self-descriptions
  (main's `selfdesc/`) for every config, then the temporal tests. Its own job database, tmux
  session and dashboard port (8767), so it can run beside a panel run.
- `runner/run.sh` — start / attach / status / stop / merge; prefix with `selfdesc` for the CPU
  job. See its header.

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

## Self-descriptions (CPU job)

Drives main's `selfdesc/` the way the panel run drives `detectors/`: main owns the science, this
branch only schedules it. Stages, all resumable:

1. **prepare**: `selfdesc.sweep --frozen --plan-json` writes `images.parquet` and checks that every
   FRE is one of those frozen on the local sample (`experiments/selfdesc-sample/fre`, tracked in
   main). It fetches the authors' released FRE if absent and **refuses to train** a new one.
2. **extract**: one chunk = one 256-row shard of one config = one `selfdesc.extract --shard S`
   subprocess (one BLAS thread, `nice 10`). `K8_M11_B11_L3` goes first, the rest cheapest-first.
3. **assemble**: each config, as soon as its shards are all on disk.
4. **temporal**: `selfdesc.temporal --suite S` (D-008/D-009/D-011) for each `--temporal-configs`
   entry, with every block cached so a stop resumes.

`--channels luma rgb` adds `_rgb` configs (Φ per colour plane); `--suite arms` is the lighter
temporal suite for main's pipeline-arm experiments.

The CPU governor allows `min(40, cores − 4 − other users' load)` workers, recomputed every 20 s.
The courtesy PAUSE file, the dashboard pause and the disk floor stop it, as for the panel.

```bash
EXP=~/anchor-date-forensics/experiments
./runner/run.sh selfdesc smoke "$EXP/stratified-36k/manifest.csv" "$EXP/selfdesc-36k"   # -> selfdesc-36k-smoke
./runner/run.sh selfdesc start "$EXP/stratified-36k/manifest.csv" "$EXP/selfdesc-36k"
./runner/run.sh selfdesc status
```

Several runs back to back (`runner/selfdesc_queue.py`): a queue file holds one
`<manifest> <out-dir> [options]` per line. The runs go in order, and the queue stops at the
first failure. Each run resumes, so restarting the queue continues where it stopped. D-011's
queue lives in main:

```bash
./runner/run.sh selfdesc queue-smoke "$EXP/selfdesc-d011.queue"   # every run as a smoke run
./runner/run.sh selfdesc queue "$EXP/selfdesc-d011.queue"
```

Dashboard: `ssh -F ssh_config -L 8767:127.0.0.1:8767 remote`, then http://localhost:8767.
Smoke runs always write to `<out-dir>-smoke`, because their `images.parquet` describes a subset.
