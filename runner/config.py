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
    # --expect-aes 2, not --require-all-aes. AEROBLADE's paper uses three autoencoders and
    # stabilityai/stable-diffusion-2-base is one of them, but as of 2026-09-25 the entire SD2
    # family is withdrawn from Hugging Face: 401 from the API, 404 in a logged-in browser, while
    # the rest of the org serves normally. There is no licence left to accept, so three is
    # unobtainable and requiring it would just never run.
    #
    # Pinning the count rather than dropping the check is the point. Bare degradation would let a
    # *second* autoencoder disappear later and still produce a plausible-looking score file;
    # asserting exactly two turns that into a failed chunk. The score is a max over autoencoders,
    # so scoring with a subset can only lower it -- conservative toward calling images authentic,
    # which is the safe direction here.
    # --recon-dir onto /data: AEROBLADE keeps every autoencoder round-trip as a PNG, and its
    # default puts them under detectors/aeroblade/ on the 492 GB root volume shared with 18 other
    # home directories. Measured at 58.7 KB each, that is 4.3 GB and 72,000 files for a 36k run,
    # 43 GB and 711,000 files for the full 355,638 -- fine on /data, rude on /.
    # The path is keyed by crop policy and dtype inside run_score.py, so fp32 and fp16
    # reconstructions cannot be mistaken for each other.
    Member("aeroblade", "aeroblade",
           ["--dtype", "fp16", "--expect-aes", "2",
            "--recon-dir", str(WORK_DIR.parent / "recon")],
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

# Hold when either volume drops below this. On 2026-09-26 an unbounded joblib cache in
# detectors/aeroblade/ filled the 492 GB root volume to zero bytes at 95% through an eight-hour
# run, which killed the run and, worse, left eighteen other people unable to write. Everything
# large is on /data now, but a job that can exhaust a shared filesystem should stop itself rather
# than rely on having predicted every path something writes to.
MIN_FREE_BYTES = int(os.environ.get("PANEL_MIN_FREE_GB", 20)) * 1_000_000_000

# --- run-level -------------------------------------------------------------------------------

MAX_ATTEMPTS = 3          # a chunk that fails this many times is left failed for a human
SUBPROCESS_TIMEOUT = 3 * 60 * 60   # a chunk that has not finished in 3h is wedged, not slow

HOST = "127.0.0.1"
PORT = int(os.environ.get("PANEL_PORT", 8766))   # 8765 is the downloader's
POLL_SECONDS = 2


_WORK_ROOT = WORK_DIR


def scope_to_manifest(manifest_sha):
    """Give each manifest its own work directory and job database.

    Chunks are keyed by (detector, seq), and seq only means anything relative to one manifest:
    chunk 0 of a 36,000-row stratified draw and chunk 0 of the full 355,638 are both
    (detector, seq 0, start 0, n 2500) while covering entirely different images. Sharing state
    between them lets one run's outputs be claimed as the other's. merge.py would catch it --
    it checks the concatenated image_ids against the manifest, in order -- but only after
    everything had been re-scored, and the end of an eight-hour run is a bad place to discover
    it. Scoping by content hash makes the collision impossible instead of merely detectable,
    and still lets the same manifest resume exactly where it stopped.
    """
    global WORK_DIR, STATE_DB
    tag = manifest_sha[:12]
    WORK_DIR = _WORK_ROOT / tag
    STATE_DB = VAR_DIR / f"scoring-{tag}.sqlite"
    return tag


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
