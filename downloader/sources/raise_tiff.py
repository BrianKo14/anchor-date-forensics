"""RAISE: uncompressed camera TIFFs -> q95 JPEG. The run's critical path.

RAISE matters out of proportion to its size. It is the only authentic source in AI-GenBench that is
not a web image, which makes it the closest available stand-in for genuinely digitized archival
material -- the thesis's central open question. It is also, by a wide margin, the slowest thing here:

    measured 2026-09-14 against 193.205.194.113
    one connection      75 KB/s
    six connections    ~365 KB/s aggregate, and two of the six stalled outright
    one TIFF           36,939,560 bytes (the 20 MB in imports/sample/authentic.py:35 is wrong)

So 1,000 images is roughly a day and the full 5,362 would be the better part of a week. The
concurrency cap in config.HARD_CONCURRENCY_CAP is politeness to a university host that has already
shown it stalls under parallel load, not a tuning parameter -- raising it makes the run slower, not
faster.

Those stalls are also why this source needs both a long read timeout and a retry pass: on the first
real run, 7 of 11 attempts failed on a 30 s timeout, none of them because the file was missing.

The subset keeps all 64 validation ids and draws the remainder from train: the validation half is
small enough to take whole, and losing it entirely would leave that split with no archival material
at all. Candidates are oversampled (config.RAISE_OVERSAMPLE) because, unlike LAION, the pinned
selection has no built-in slack -- without it every permanent failure is one image below target.
"""

import random

import config  # noqa: F401 -- must precede common/aigenbench: it sets AIGENBENCH_DATA_ROOT
import aigenbench
import fetch
from common import AUTHENTIC_DIR, image_path
from imaging import decode_and_validate, prepare_image
from sources import pump

NAME = "raise"
SUBSET_SEED = 1234

# A smoke run takes three images, not --limit's twenty. At 36.9 MB and ~350 KB/s each, twenty would
# be a 35-minute "quick rehearsal"; three is about five minutes and exercises exactly the same path.
SMOKE_CAP = 3


def catalog():
    """{stem: TIFF url} from the CSV shipped in the repo.

    The CSV is the only way to get these URLs -- RAISE's catalogue sits behind a confirmation form
    at loki.disi.unitn.it, and the images themselves are served from a bare IP.
    """
    if not config.RAISE_CSV.exists():
        raise SystemExit(
            f"{config.RAISE_CSV} not found -- RAISE cannot be planned.\n"
            f"Get it from {config.RAISE_SOURCE} ('Get the images!') and save it there."
        )
    import pandas as pd

    frame = pd.read_csv(config.RAISE_CSV)
    return dict(zip(frame["File"], frame["TIFF"]))


def select(verbose=True):
    """The (file_id, split, url) rows we intend to fetch, deterministically chosen."""
    urls = catalog()
    chosen = []

    # Validation first and in full: 64 ids against train's 5,298.
    for split in ("validation", "train"):
        pinned = [fid for fid in aigenbench.real_file_ids(split, verbose=False)
                  if fid.startswith("RAISE/")]
        available = [fid for fid in pinned if fid.split("/", 1)[1] in urls]

        if split == "validation":
            picked = sorted(available)
        else:
            # Draw more candidates than the target so permanent failures can be absorbed; the run
            # stops at RAISE_TARGET successes, so the extras cost nothing unless they are needed.
            budget = int(config.RAISE_TARGET * config.RAISE_OVERSAMPLE)
            room = max(0, budget - len(chosen))
            # Shuffle before truncating: the CSV is ordered by capture session, so taking the first
            # N would give a subset of one afternoon's photographs rather than of RAISE.
            shuffled = sorted(available)
            random.Random(SUBSET_SEED).shuffle(shuffled)
            picked = shuffled[:room]

        chosen += [(fid, split, urls[fid.split("/", 1)[1]]) for fid in picked]

    if verbose:
        by_split = {}
        for _, split, _ in chosen:
            by_split[split] = by_split.get(split, 0) + 1
        print(f"RAISE: {len(chosen)} candidates drawn for a target of {config.RAISE_TARGET} "
              f"(of 5,362 pinned) {by_split}")
        print(f"  ~{config.RAISE_TARGET * 36.9 / 1000:.1f} GB of traffic, "
              f"~{config.RAISE_TARGET * 36.9e6 / 400e3 / 3600:.0f} h at ~400 KB/s")
    return chosen


def plan(ctx):
    rows = []
    for file_id, split, url in select():
        dest = image_path(AUTHENTIC_DIR, file_id)
        rows.append({
            "source": NAME,
            "split": split,
            "file_id": file_id,
            "url": url,
            "dest": str(dest.relative_to(config.DATA_ROOT)),
        })
    if ctx.smoke:
        rows = rows[:SMOKE_CAP]
        ctx.state.add_items(rows)
        return len(rows)

    ctx.state.add_items(rows)
    # The dashboard total is the target, not the oversampled candidate count -- otherwise the bar
    # would stop at 71% on a completely successful run.
    return config.RAISE_TARGET


def handle_one(ctx, item):
    dest = config.DATA_ROOT / item["dest"]
    if dest.exists():
        return "done", dest.stat().st_size, None, "already on disk"

    raw, reason = fetch.fetch_bytes(item["url"], bucket=ctx.bucket, stop=ctx.stop,
                                    read_timeout=config.RAISE_READ_TIMEOUT,
                                    on_progress=ctx.progress(NAME))
    if raw is None:
        return ("pending" if reason == "interrupted" else "failed"), 0, None, reason

    image = decode_and_validate(raw)
    if image is None:
        return "failed", 0, None, "undecodable or under 200px"

    data = prepare_image(image)
    sha = fetch.write_atomic(dest, data)
    # Book the *downloaded* bytes, not the written ones: the 36.9 MB TIFF is what cost time, and
    # the dashboard's ETA is only useful if it reflects the traffic.
    return "done", len(raw), sha, "ok"


def reached_target(ctx):
    """False once RAISE_TARGET images are on disk -- stops the oversampled queue early."""
    if ctx.smoke:
        return ctx.state.counts().get(NAME, {}).get("done", 0) < SMOKE_CAP
    return ctx.state.counts().get(NAME, {}).get("done", 0) < config.RAISE_TARGET


RETRY_ROUNDS = 2


def run(ctx):
    done, failed = pump(ctx, NAME, handle_one, batch=8, until=reached_target, records_bytes=False)

    # Same reasoning as LAION's retry pass, and more important here: a RAISE timeout is almost
    # always the host stalling, not the file being gone, and every unrecovered failure is one fewer
    # archival image for the false-positive experiment this subset exists to support.
    for round_number in range(RETRY_ROUNDS):
        if ctx.stop.is_set() or not reached_target(ctx):
            break
        requeued = ctx.state.retry_transient(NAME)
        if not requeued:
            break
        ctx.log("info", f"retrying {requeued} transient failures "
                        f"(round {round_number + 1}/{RETRY_ROUNDS})", NAME)
        extra_done, extra_failed = pump(ctx, NAME, handle_one, batch=8, until=reached_target, records_bytes=False)
        done += extra_done
        failed += extra_failed

    return done, failed
