"""Paths, targets and defaults for the full-dataset download.

Every number here was measured against the lab server on 2026-09-14 rather than assumed; the
comments record what was measured so a later reader can tell a deliberate choice from a guess.

The main branch's `imports/sample/` importers build a 612+612 image miniature into `data/`. This
package builds the real thing -- 180,000 real + 180,000 fake, both splits -- into a separate root.
`common.py`, `aigenbench.py` and `imaging.py` are copied in flat alongside the rest of this package
(not imported from `imports/sample/`, which doesn't exist on this branch) so the two datasets stay
byte-compatible; pipeline.py/verify.py put this directory on sys.path, which is what makes their
flat `from common import ...` style imports resolve.
"""

import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# Where the bytes land. /data/aigenbench on the server: /data has 1.5 TB free, the root filesystem
# only 110 GB and it is shared with 18 other users. Set before importing common, which reads it.
DATA_ROOT = Path(os.environ.setdefault("AIGENBENCH_DATA_ROOT", "/data/aigenbench"))

# Everything that is not downloaded bytes stays in $HOME, per the run's ground rules.
VAR_DIR = Path(os.environ.get("AIGENBENCH_VAR_DIR") or PROJECT_ROOT / "var")
STATE_DB = VAR_DIR / "downloads.sqlite"
LOG_DIR = VAR_DIR / "logs"

AUTHENTIC_DIR = DATA_ROOT / "authentic"
FAKES_DIR = DATA_ROOT / "fakes"
CACHE_DIR = DATA_ROOT / "cache"
PARQUET_CACHE = CACHE_DIR / "parquet"
ZIP_CACHE = CACHE_DIR / "zips"

# Courtesy brake for the other eighteen people on this machine. The dashboard binds to localhost and
# the job runs as one user, so a colleague who finds it saturating the uplink at 3am has no way to
# ask it to stop short of killing a multi-day job. Creating this file pauses it; deleting it
# resumes. The directory is world-writable (sticky) precisely so that anyone can.
CONTROL_DIR = DATA_ROOT / "control"
PAUSE_FILE = CONTROL_DIR / "PAUSE"

SPLITS = ("train", "validation")

SOURCES = ("raise", "fakes", "coco", "laion")

# RAISE first and alone at the head of the list: at ~350 KB/s aggregate for 36.9 MB TIFFs it is
# ~29 h of work, and every other source finishes inside it. Starting it last would serialise the run.
SOURCE_ORDER = ("raise", "fakes", "coco", "laion")

# --- targets -----------------------------------------------------------------------------------
#
# Real-image counts are not configurable: they are AI-GenBench's pinned selection, read from the
# cloned repo's file-id lists. RAISE is the one exception -- see below.

# Of the 5,362 RAISE ids the benchmark pins, we take a subset. The full set is 198 GB of traffic
# against a 75 KB/s host in Italy, i.e. ~6.5 days; 1,000 is ~29 h and still gives a false-positive
# rate on digitized material to +-0.37% at the 95% level when the observed rate is 0.
RAISE_TARGET = int(os.environ.get("AIGENBENCH_RAISE_TARGET", 1000))

# Per-generator row counts in the published fake part, asserted after extraction.
FAKE_ROWS_PER_GENERATOR = {"train": 4000, "validation": 1000}
N_GENERATORS = 36

# --- sources -----------------------------------------------------------------------------------

# The S3 path-style endpoint, not the images.cocodataset.org vanity host: that host is a CNAME onto
# the same bucket but serves an *.s3.amazonaws.com certificate, so HTTPS fails hostname validation.
# (Same reasoning as imports/sample/authentic.py:40.)
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ZIPS = {
    # name: (url suffix, exact Content-Length verified 2026-09-14, md5 or None)
    # The two image zips carry multipart ETags ("...-98", "...-2306") which are *not* MD5s of the
    # content, so they get size + CRC via ZipFile.testzip() instead. The annotations ETag has no
    # dash and is a real MD5.
    "val2017": ("zips/val2017.zip", 815585330, None),
    "train2017": ("zips/train2017.zip", 19336861798, None),
    "annotations": ("annotations/annotations_trainval2017.zip", 252907541,
                    "f4bbac642086de4f52a3fdda2de5fa2c"),
}
COCO_DIRS = {"COCO2017_train": "train2017", "COCO2017_val": "val2017"}

# RAISE serves from a bare IP; loki.disi.unitn.it is the catalogue host, 193.205.194.113 the files.
RAISE_CSV = PROJECT_ROOT / "RAISE_urls.csv"
RAISE_SOURCE = "http://loki.disi.unitn.it/RAISE/confirm.php?package=all"

HF_REPO = "lrzpellegrini/AI-GenBench-fake_part"
HF_SHARDS = {"train": 57, "validation": 15}

# --- rate limiting -----------------------------------------------------------------------------

# Measured ceilings: HF ~8 MB/s at 4 connections, COCO ~1.9 MB/s, RAISE ~365 KB/s aggregate at 6
# connections (2 of which stalled outright). The defaults sit just under each so we are not the
# reason a link saturates.
DEFAULT_RATE_BYTES = int(os.environ.get("AIGENBENCH_RATE_BYTES", 12_000_000))

DEFAULT_CONCURRENCY = {
    "raise": 4,
    "fakes": 4,      # parallel byte ranges within the shard being fetched.
    "coco": 6,       # likewise. Only two files exist, but a single stream to S3 measured 204 KB/s
                     # while a concurrent request pulled another 278 KB/s, so the per-connection
                     # rate -- not the link -- is the limit. Segments are how COCO goes faster.
    "laion": 16,     # latency-bound, tiny files, and a fifth of the URLs are dead and must time out.
}

# RAISE's cap is not a tuning knob. loki.disi.unitn.it is a university host that gave us 75 KB/s on
# one connection and stalled 2 of 6 parallel ones; hammering it harder is rude and counterproductive.
HARD_CONCURRENCY_CAP = {"raise": 4}

MAX_CONCURRENCY = 32

# --- load governor -----------------------------------------------------------------------------
#
# The box has 48 cores and sat at load ~4.1 with two other users' jobs running when this was
# written. Fetching is I/O-bound and barely registers, but decoding and re-encoding 360,000 images
# at JPEG q95 is genuinely CPU-heavy, so the two pools are governed separately.

NPROC = os.cpu_count() or 48
GOVERNOR_INTERVAL = 15.0
LOAD_PAUSE = NPROC * 1.00      # somebody else needs the machine -- stop fetching
LOAD_THROTTLE = NPROC * 0.75   # halve both pools
LOAD_RESTORE = NPROC * 0.40    # back to configured levels

DEFAULT_CPU_WORKERS = max(2, min(8, NPROC // 6))

# --- http --------------------------------------------------------------------------------------

# Identifies the client without identifying the person. This header goes to every host we touch --
# HuggingFace, S3, RAISE, and a few thousand assorted LAION domains of unknown provenance -- so it
# carries no address, no username, and nothing else worth harvesting.
USER_AGENT = "anchor-date-forensics/0.1 (academic dataset download; FCEN-UBA)"
CONNECT_TIMEOUT = 10
READ_TIMEOUT = 30          # HF docs recommend 30 on slow links; RAISE needs every second of it.
MAX_ATTEMPTS = 4
BACKOFF_BASE = 2.0
CHUNK_BYTES = 1 << 20

# LAION link rot is expected, but "dead" and "slow" are different things and the first smoke run
# conflated them: at a 10 s read timeout, 65 of 102 failures were timeouts against only 26 genuine
# 403/404/410/connection errors. A truly unreachable host still fails fast on CONNECT_TIMEOUT; this
# only governs how long we wait for a host that has already answered. 25 s converts most of those
# timeouts into images, which matters because validation LAION has just 1.9x spare capacity.
LAION_READ_TIMEOUT = 25

# RAISE needs the opposite treatment. A 36.9 MB TIFF at the measured 75-500 KB/s is a 1-8 minute
# transfer, and the host demonstrably stalls mid-stream -- two of six parallel connections froze
# outright in testing. At a 30 s read timeout that showed up as a 64% failure rate. This is a
# per-read timeout, not a total deadline, so a long value costs nothing on a healthy transfer.
RAISE_READ_TIMEOUT = 120

# And unlike LAION, RAISE has no natural oversampling: the selection picks exactly RAISE_TARGET
# ids, so any permanent failure lands one image below target. Draw extra candidates and stop at
# the target instead.
RAISE_OVERSAMPLE = 1.4

# --- server ------------------------------------------------------------------------------------

# 127.0.0.1 only, never 0.0.0.0. This is a shared machine with 18+ other home directories and the
# dashboard exposes write controls. Reach it with:
#     ssh -F ssh_config -L 8765:127.0.0.1:8765 remote
HOST = "127.0.0.1"
PORT = int(os.environ.get("AIGENBENCH_PORT", 8765))

POLL_SECONDS = 2


def ensure_dirs():
    for path in (VAR_DIR, LOG_DIR, AUTHENTIC_DIR / "images", FAKES_DIR / "images",
                 PARQUET_CACHE, ZIP_CACHE):
        path.mkdir(parents=True, exist_ok=True)

    # 1777: anyone may drop a PAUSE file, but only its owner may remove someone else's -- the same
    # arrangement /tmp uses, for the same reason.
    CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    try:
        CONTROL_DIR.chmod(0o1777)
    except OSError:
        pass  # not ours to chmod; the brake still works for anyone who can write here


def install_notes():
    """Put the explanatory READMEs where someone would actually look for them.

    Copied from the repo on every start rather than written once, so editing the source updates the
    deployed copy and the two cannot silently diverge. Two destinations because there are two ways
    to stumble onto this: finding the data directory, or finding the process.
    """
    written = []
    for source_name, destination in (
        ("data_readme.md", DATA_ROOT / "README.md"),
        ("home_readme.md", Path.home() / "README.md"),
    ):
        source = PROJECT_ROOT / "downloader" / source_name
        if not source.exists():
            continue
        try:
            text = source.read_text()
            if not destination.exists() or destination.read_text() != text:
                destination.write_text(text)
                written.append(str(destination))
        except OSError as error:
            print(f"could not write {destination}: {error}")
    return written


def describe():
    """One-screen summary of where everything will go, printed at startup."""
    return "\n".join([
        f"data root   {DATA_ROOT}",
        f"state       {STATE_DB}",
        f"logs        {LOG_DIR}",
        f"dashboard   http://{HOST}:{PORT}",
        f"raise target {RAISE_TARGET} of 5362 pinned ids",
    ])
