"""LAION-400M: scraped web URLs -> q95 JPEG. The source that is allowed to fail.

The benchmark ships the scraped URLs in `{split}_laion400m_filelist.json` -- 142,018 entries for the
55,618 train images it needs, 45,462 for validation's 23,862. That 2.4x headroom is not incidental;
it is there because these are ordinary web URLs that rot. A 120-URL sample measured 79.2% still
alive, which is comfortably enough: 142,018 x 0.79 is about twice the train target.

Because the queue is oversampled, this is the one source that stops on a *count* rather than on an
empty queue. Everything not reached is left pending, which is also what makes the target adjustable
later without re-planning.

Two things follow from the URLs being shuffled before insertion:
  - stopping at the first N successes stays unbiased, and
  - the spare images absorb rot without correlating the survivors with any one host or crawl date.
"""

import functools
import random

import config  # noqa: F401 -- must precede common/aigenbench: it sets AIGENBENCH_DATA_ROOT
import aigenbench
import fetch
from common import AUTHENTIC_DIR, image_path
from imaging import decode_and_validate, prepare_image
from sources import pump

NAME = "laion"
SHUFFLE_SEED = 1234


@functools.lru_cache(maxsize=1)
def targets():
    """{split: how many LAION images the benchmark pins for it}.

    Cached because `enough()` consults it once per batch and the underlying id list is 144,000
    lines of text -- re-reading it every few seconds for thirty hours would be absurd.
    """
    out = {}
    for split in config.SPLITS:
        ids = aigenbench.real_file_ids(split, verbose=False)
        out[split] = sum(1 for fid in ids if fid.startswith("LAION-400M/"))
    return out


def plan(ctx):
    wanted = targets()
    rows = []

    for split in config.SPLITS:
        entries = aigenbench.laion_filelist(split)
        # Shuffle before truncating so the oversampling stays unbiased -- same reasoning as
        # imports/sample/authentic.py::harvest, which this generalises.
        entries = list(entries)
        random.Random(f"{SHUFFLE_SEED}:{split}").shuffle(entries)

        need = ctx.target_count(wanted[split])
        # Enqueue the whole filelist, not a computed multiple of the target. A queue row costs a few
        # bytes and `enough()` stops the run the moment the target is met, so there is no reason to
        # bet on a survival rate in advance -- and betting wrong on validation, which ships only 1.9x
        # spare capacity, would mean discovering the shortfall hours in.
        enqueue = len(entries) if not ctx.smoke else min(len(entries), need * 4)

        for entry in entries[:enqueue]:
            file_id = f"LAION-400M/{entry['id']}"
            rows.append({
                "source": NAME,
                "split": split,
                "file_id": file_id,
                "url": entry["url"],
                "dest": str(image_path(AUTHENTIC_DIR, file_id).relative_to(config.DATA_ROOT)),
            })
        print(f"LAION {split}: target {need}, enqueued {enqueue} of {len(entries)} available")

    ctx.state.add_items(rows)
    # The *target*, not the queue length: the queue is oversampled on purpose, so reporting its
    # size would show a progress bar that can never reach the end.
    return sum(ctx.target_count(n) for n in wanted.values())


def enough(ctx):
    """True while any split is still short of its target -- the early-stop predicate for pump()."""
    counts = ctx.state.done_by_split(NAME)
    return any(counts.get(split, 0) < ctx.target_count(n) for split, n in targets().items())


def short_splits(ctx):
    """The splits still short of target -- keeps claim() from draining an already-met split's
    oversampled backlog just because a sibling split (enqueued later, with higher ids) still needs
    work. See state.claim()'s docstring."""
    counts = ctx.state.done_by_split(NAME)
    return [split for split, n in targets().items() if counts.get(split, 0) < ctx.target_count(n)]


def handle_one(ctx, item):
    dest = config.DATA_ROOT / item["dest"]
    if dest.exists():
        return "done", dest.stat().st_size, None, "already on disk"

    # LAION_READ_TIMEOUT, not the global one: see config for why 25 s rather than the 10 s this
    # started at. Failures here are ordinary and get tallied, not raised.
    raw, reason = fetch.fetch_bytes(item["url"], bucket=ctx.bucket, stop=ctx.stop,
                                    read_timeout=config.LAION_READ_TIMEOUT,
                                    on_progress=ctx.progress(NAME))
    if raw is None:
        return ("pending" if reason == "interrupted" else "failed"), 0, None, reason

    image = decode_and_validate(raw)
    if image is None:
        return "failed", 0, None, "undecodable or under 200px"

    data = prepare_image(image)
    sha = fetch.write_atomic(dest, data)
    return "done", len(raw), sha, "ok"


# Extra passes over the transient failures, once the queue has drained and a split is still short.
# Bounded rather than open-ended: a URL that times out three times is, for our purposes, dead.
RETRY_ROUNDS = 2


def run(ctx):
    done, failed = pump(ctx, NAME, handle_one, batch=64, until=enough, records_bytes=False,
                        needed_splits=short_splits)

    for round_number in range(RETRY_ROUNDS):
        if ctx.stop.is_set() or not enough(ctx):
            break
        requeued = ctx.state.retry_transient(NAME)
        if not requeued:
            break
        short = {split: ctx.target_count(n) - ctx.state.done_by_split(NAME).get(split, 0)
                 for split, n in targets().items()}
        ctx.log("info", f"still short {({k: v for k, v in short.items() if v > 0})}; "
                        f"retrying {requeued:,} transient failures "
                        f"(round {round_number + 1}/{RETRY_ROUNDS})", NAME)
        extra_done, extra_failed = pump(ctx, NAME, handle_one, batch=64, until=enough,
                                        records_bytes=False, needed_splits=short_splits)
        done += extra_done
        failed += extra_failed

    return done, failed
