"""One module per origin dataset, plus the worker pump they share.

Each source module exposes two functions:

    plan(ctx)   idempotent; writes the rows it intends to fetch into the items/artifacts tables
    run(ctx)    works the queue until it is empty or the run is stopped

They do *not* share a per-item interface, because the four sources genuinely differ in shape: COCO
and the fake part arrive as a handful of large archives that are then unpacked locally, while LAION
and RAISE are tens of thousands of individual HTTP GETs. Forcing one abstraction over both would
mean pretending a 19 GB zip is an image.

What they do share is `pump`, below: the concurrent claim-fetch-write loop, which is the part that
has to get pausing, throttling and resume right.
"""

from concurrent.futures import ThreadPoolExecutor

import config


def cpu_workers(ctx):
    """Pool size for local decode/re-encode work, governed by load rather than by politeness.

    Unpacking COCO's zips and the fake part's parquet shards is pure CPU -- no remote host cares how
    many threads we use, but the other people on this 48-core box do.
    """
    try:
        want = int(ctx.state.get_control("cpu_workers", config.DEFAULT_CPU_WORKERS))
    except (TypeError, ValueError):
        want = config.DEFAULT_CPU_WORKERS
    return max(0, ctx.governor.scale(max(1, min(want, config.MAX_CONCURRENCY))))


def pump(ctx, source, handle_one, batch=32, until=None, workers_of=None, records_bytes=True,
         needed_splits=None):
    """Work `source`'s pending items concurrently until the queue drains or the run stops.

    `handle_one(ctx, item)` returns (status, bytes, sha256, reason) and must be safe to call from
    several threads. It is expected to be idempotent -- if the destination already exists it should
    report done without refetching, because `pump` will happily hand it a row that a previous run
    had already written but not recorded.

    `until(ctx)` is an optional early-stop predicate, returning False when enough work is done. LAION
    needs it: its queue is deliberately oversampled 2.4x to absorb link rot, so draining the queue
    would fetch 60,000 images nobody asked for.

    `needed_splits(ctx)`, when given, is consulted every batch and passed to `claim` so a split that
    has already met its own target stops being claimed even while `until` says to keep going overall
    (true for LAION: train and validation have independent targets, and claiming by id alone would
    keep draining whichever split was enqueued first long after it stopped needing more).

    Concurrency is re-read from the control table every batch rather than fixed at pool creation, so
    moving the slider in the dashboard takes effect within one batch instead of requiring a restart.
    """
    state = ctx.state
    done = failed = 0

    while not ctx.stop.is_set():
        if until is not None and not until(ctx):
            break
        if not ctx.wait_while_paused(source):
            break
        ctx.sync_rate()

        workers = workers_of(ctx) if workers_of else ctx.concurrency(source)
        if workers <= 0:
            if ctx.stop.wait(2.0):
                break
            continue

        splits = needed_splits(ctx) if needed_splits else None
        if needed_splits is not None and not splits:
            break
        items = state.claim(source, limit=max(batch, workers), splits=splits)
        if not items:
            break

        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=source) as pool:
            results = list(pool.map(lambda item: _guarded(ctx, source, handle_one, item), items))

        for item, (status, size, sha, reason) in zip(items, results):
            if status == "done":
                done += 1
                state.finish(item["id"], "done", bytes_=size, sha256=sha)
                # Sources that stream progress per chunk have already booked these bytes; booking
                # them again here would double the reported throughput.
                if size and records_bytes:
                    state.record_bytes(source, size)
            elif status == "pending":
                # Interrupted rather than failed: hand it back so the next run retries it cleanly.
                state.release(item["id"], error=reason)
            else:
                failed += 1
                state.finish(item["id"], status, bytes_=size, sha256=sha, error=reason)

    return done, failed


def _guarded(ctx, source, handle_one, item):
    if ctx.stop.is_set():
        return "pending", 0, None, "interrupted"
    try:
        return handle_one(ctx, item)
    except Exception as error:  # noqa: BLE001 -- one bad image must not take the run down
        return "failed", 0, None, f"{type(error).__name__}: {error}"[:200]
