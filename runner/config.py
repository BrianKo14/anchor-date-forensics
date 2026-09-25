"""Panel definition, paths and defaults for an unattended scoring run.

Adapted from the `downloader` branch's config.py, which did the same job for the 360k download:
one module holding every number, with the reasoning next to it, so a later reader can tell a
measured choice from a guess.

The work unit here is a *chunk* -- a contiguous slice of the manifest scored by one detector in
one subprocess. Chunking rather than one long process per detector is what makes the run
resumable: a chunk that completes is never redone, and a run killed at 3am picks up where it
stopped instead of losing the night.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

VAR_DIR = Path(os.environ.get("PANEL_VAR_DIR") or PROJECT_ROOT / "var")
STATE_DB = VAR_DIR / "scoring.sqlite"
LOG_DIR = VAR_DIR / "logs"

# Chunk manifests and per-chunk score CSVs. Thousands of small files, and they are scratch --
# merge.py turns them into the two artifacts that matter. Kept beside the data rather than in the
# repo so a re-run does not churn the working tree.
WORK_DIR = Path(os.environ.get("PANEL_WORK_DIR") or "/data/aigenbench/panel/work")

# Courtesy brake, same contract as the downloader's: anyone on this shared machine can create
# this file to make the run stand down, and remove it to let it continue. Directory is 1777.
CONTROL_DIR = Path(os.environ.get("PANEL_CONTROL_DIR") or "/data/aigenbench/control")
PAUSE_FILE = CONTROL_DIR / "PAUSE"


class Member:
    """One panel member: how to invoke it, and how much work to hand it at a time."""

    def __init__(self, name, directory, extra=(), chunk=2500, batch=64):
        self.name = name
        self.directory = directory
        self.extra = list(extra)
        self.chunk = chunk
        self.batch = batch

    @property
    def python(self):
        return PROJECT_ROOT / "detectors" / self.directory / ".venv" / "bin" / "python"

    @property
    def script(self):
        return PROJECT_ROOT / "detectors" / self.directory / "run_score.py"

    def __repr__(self):
        return f"Member({self.name})"


# Ordered cheapest-first, measured on the 1224-image sample (seconds per image, CPU):
# cnndetection 0.24, univfd 0.51, dmimagedetection 2.38/2.40, aeroblade 6.23. Cheapest-first
# means a run that dies overnight still leaves three complete score files rather than five
# partial ones, and the first numbers to look at in the morning arrive soonest.
#
# aeroblade gets --dtype fp16: upstream's own precision, restored now that there is a CUDA device
# to run it on. It is also the memory-heaviest member (three autoencoders), hence the smaller
# batch.
PANEL = [
    Member("cnndetection", "cnndetection"),
    Member("univfd", "univfd"),
    # batch 16, not the default 64: this network is fully convolutional and its score is the mean
    # of a spatial logit map, so activations are far larger than a classifier's and 64 OOMs the
    # 10 GB card. Measured 2026-09-25: batches of 8, 16 and 32 all score 1,000 images in 11.7 s,
    # so the smaller batch is free -- it is compute-bound, not batch-bound.
    Member("dmimagedetection_progan", "dmimagedetection", ["--model", "Grag2021_progan"], batch=16),
    Member("dmimagedetection_latent", "dmimagedetection", ["--model", "Grag2021_latent"], batch=16),
    # --require-all-aes because this run is configured to match the paper. Without a Hugging Face
    # token for the gated stabilityai/stable-diffusion-2-base repo, run_score.py would otherwise
    # degrade to two of three autoencoders, print a warning nobody reads at 3am, and produce a
    # score file that looks exactly like a good one. Failing the chunk is the honest outcome.
    Member("aeroblade", "aeroblade", ["--dtype", "fp16", "--require-all-aes"],
           chunk=2500, batch=32),
]

PANEL_BY_NAME = {m.name: m for m in PANEL}
PANEL_ORDER = [m.name for m in PANEL]

# --- devices -------------------------------------------------------------------------------

# sapucay has two RTX 3080s (12 GB and 10 GB). One worker per GPU, pinned with
# CUDA_VISIBLE_DEVICES rather than --device cuda:N, so that any .cuda() inside upstream code --
# aeroblade's distance module in particular -- lands on the intended card instead of card 0.
GPUS = [int(g) for g in os.environ.get("PANEL_GPUS", "0,1").split(",") if g != ""]

# --- governor ------------------------------------------------------------------------------
#
# Shared machine, 18+ other home directories. The run must yield rather than win.

NPROC = os.cpu_count() or 48
GOVERNOR_INTERVAL = 20.0

# Someone else's compute process on a GPU we were about to use: stop claiming for that GPU and
# wait. This is the etiquette rule that matters most -- a 3080 with 10 GB has room for exactly
# one serious job, so sharing a card means both jobs thrash.
YIELD_TO_OTHER_GPU_PROCS = os.environ.get("PANEL_YIELD_GPU", "1") != "0"

# Load average above this and we hold. Scoring is GPU-bound but the dataloader side is not free,
# and the box idles around 4.
LOAD_PAUSE = NPROC * 0.90

# --- run-level -------------------------------------------------------------------------------

MAX_ATTEMPTS = 3          # a chunk that fails this many times is left failed for a human
SUBPROCESS_TIMEOUT = 3 * 60 * 60   # a chunk that has not finished in 3h is wedged, not slow

HOST = "127.0.0.1"
PORT = int(os.environ.get("PANEL_PORT", 8766))   # 8765 is the downloader's
POLL_SECONDS = 2


def ensure_dirs():
    for path in (VAR_DIR, LOG_DIR, WORK_DIR, WORK_DIR / "chunks", WORK_DIR / "scores"):
        path.mkdir(parents=True, exist_ok=True)
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CONTROL_DIR.chmod(0o1777)
    except OSError:
        pass  # not ours to chmod; the brake still works for anyone who can write there


def describe(manifest, out_dir):
    return "\n".join([
        f"manifest   {manifest}",
        f"outputs    {out_dir}",
        f"work       {WORK_DIR}",
        f"state      {STATE_DB}",
        f"logs       {LOG_DIR}",
        f"gpus       {GPUS}",
        f"panel      {', '.join(PANEL_ORDER)}",
        f"dashboard  http://{HOST}:{PORT}",
    ])
