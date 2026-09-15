"""The run context handed to every source: shared state, throttles, and the stop signal.

Pulling this out of pipeline.py keeps the source modules importable on their own (useful when
poking at one source in a REPL) and avoids a circular import between the orchestrator and the
sources it orchestrates.
"""

import threading
import time

import config
import ratelimit


class Context:
    def __init__(self, state, governor=None, bucket=None, smoke=False, limit=None):
        self.state = state
        self.bucket = bucket or ratelimit.TokenBucket(state.rate_bytes())
        self.governor = governor or ratelimit.Governor(state)
        self.stop = threading.Event()
        self.smoke = smoke
        self.limit = limit
        self.started = time.time()
        self._rate_seen = self.bucket.rate
        self._pause_cache = {}
        self._courtesy_seen = False

    # -- throttles -------------------------------------------------------------------------------

    def sync_rate(self):
        """Push a rate change made in the dashboard into the live token bucket."""
        want = self.state.rate_bytes()
        if want != self._rate_seen:
            self.bucket.set_rate(want)
            self._rate_seen = want

    def concurrency(self, source):
        """Configured concurrency, scaled by the load governor. 0 means 'stand down for now'."""
        return self.governor.scale(self.state.concurrency(source))

    def courtesy_paused(self):
        """True while someone has dropped a PAUSE file in the shared control directory.

        This is the one control a colleague on the shared machine can actually reach -- see
        config.PAUSE_FILE. Checked alongside the operator's own pause so it behaves identically:
        the job holds, keeps its partial downloads, and resumes when the file is removed.
        """
        try:
            present = config.PAUSE_FILE.exists()
        except OSError:
            return False

        # Log the transition, so the reason a 20-hour job stalled at 4am is in the log rather than
        # a mystery. Whoever asked for the brake deserves the credit in the record.
        if present != self._courtesy_seen:
            self._courtesy_seen = present
            self.state.log(
                "warn" if present else "info",
                f"courtesy PAUSE file {'created' if present else 'removed'}"
                f" ({config.PAUSE_FILE}) -- {'holding' if present else 'resuming'}",
                source="shared",
            )
        return present

    def paused(self, source):
        return (self.state.is_paused(source)
                or self.courtesy_paused()
                or not self.governor.should_fetch())

    def paused_cached(self, source, ttl=0.5):
        """`paused` with a short TTL, for callers that ask once per network chunk.

        The uncached version costs two SQLite reads; a 19 GB download over six segments would ask
        several times a second for hours. Half a second of staleness is invisible next to the 2 s
        dashboard poll.
        """
        now = time.time()
        cached = self._pause_cache.get(source)
        if cached is None or now - cached[0] > ttl:
            cached = (now, self.paused(source))
            self._pause_cache[source] = cached
        return cached[1]

    def gate(self, source, poll=0.5):
        """A per-chunk callback that holds a transfer while paused. False means the run is stopping.

        Large artifacts need this. Without it, pausing during COCO's 19 GB zip only takes effect
        once the file finishes -- hours later -- which is not a pause button in any useful sense.
        Note it *holds* rather than aborting: the connection stays open, so resuming does not cost
        a reconnect or lose the segment's position.
        """
        def hold():
            while self.paused_cached(source, ttl=poll):
                if self.stop.wait(poll):
                    return False
            return not self.stop.is_set()

        return hold

    def wait_while_paused(self, source, poll=1.0):
        """Block while paused. Returns False if the run is stopping, True if it is time to work.

        Polling the control table rather than using a condition variable is deliberate: the
        dashboard writes the table from a different process's connection, so there is nothing to
        signal on. A one-second poll is well inside the 2 s the UI promises.
        """
        while self.paused(source):
            if self.stop.wait(poll):
                return False
        return not self.stop.is_set()

    # -- progress --------------------------------------------------------------------------------

    def progress(self, source, flush_bytes=8 << 20, flush_seconds=2.0):
        """A callback for fetch.download_artifact that books bytes against a source.

        Accumulates and flushes in batches. The naive version -- one INSERT plus COMMIT per 1 MB
        chunk -- costs 19,000 transactions for COCO's train zip alone, all of it to feed a
        throughput graph that is read once every two seconds.
        """
        pending = [0, time.time()]
        lock = threading.Lock()

        def record(n):
            with lock:
                pending[0] += n
                now = time.time()
                if pending[0] >= flush_bytes or now - pending[1] >= flush_seconds:
                    self.state.record_bytes(source, pending[0])
                    pending[0], pending[1] = 0, now

        return record

    def target_count(self, natural):
        """Smoke runs cap every source at --limit so the whole path can be exercised in minutes."""
        if self.smoke and self.limit:
            return min(natural, self.limit)
        return natural

    def log(self, level, msg, source=None):
        self.state.log(level, msg, source=source)

    @property
    def elapsed(self):
        return time.time() - self.started
