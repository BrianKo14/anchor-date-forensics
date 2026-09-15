"""Bandwidth throttling, per-source concurrency, and the shared-machine load governor.

Two independent controls, because the two bottlenecks are unrelated:

  the token bucket   caps *network* bytes/s. Downloading is I/O-bound and barely touches the CPU,
                     so this exists to be polite to the remote hosts and to the lab's uplink.
  the governor       caps *CPU*. Decoding and re-encoding 360,000 images at JPEG q95 is genuinely
                     expensive, and sapucay is a shared 48-core box with other people's jobs on it.

Conflating them would mean throttling the network because somebody else started a training run,
which helps nobody.
"""

import threading
import time

import config


class TokenBucket:
    """Classic token bucket over bytes. `consume` blocks until the bytes are affordable.

    Capacity is one second's worth of tokens *or one chunk, whichever is larger*. That floor is not
    cosmetic: with capacity == rate, any limit below the 1 MB chunk size makes `tokens >= n`
    unsatisfiable and the bucket deadlocks at zero throughput rather than throttling. Setting
    500 KB/s did exactly that before this floor existed.

    Holding a whole chunk's worth of capacity lets an idle worker burst by at most one chunk, which
    is the smallest burst that can make progress at all. A rate of 0 means unlimited, checked before
    any arithmetic so the disabled case costs nothing.
    """

    def __init__(self, rate_bytes):
        self._lock = threading.Lock()
        self._rate = float(rate_bytes)
        self._tokens = float(rate_bytes)
        self._updated = time.monotonic()

    def _capacity(self):
        return max(self._rate, float(config.CHUNK_BYTES))

    def set_rate(self, rate_bytes):
        with self._lock:
            self._rate = float(max(0, rate_bytes))
            self._tokens = min(self._tokens, self._capacity())

    @property
    def rate(self):
        with self._lock:
            return self._rate

    def consume(self, n, stop=None):
        """Block until `n` bytes are affordable. Returns False if `stop` fired while waiting."""
        while True:
            with self._lock:
                if self._rate <= 0:
                    return True
                now = time.monotonic()
                self._tokens = min(self._capacity(),
                                   self._tokens + (now - self._updated) * self._rate)
                self._updated = now
                # A request larger than the bucket can ever hold would wait forever; let it through
                # once the bucket is full and account for the overdraft honestly.
                affordable = min(n, self._capacity())
                if self._tokens >= affordable:
                    self._tokens -= n
                    return True
                # Never wait more than a second at a time: the rate can change underneath us when
                # the operator moves the slider, and a long sleep would ignore it.
                deficit = affordable - self._tokens
                wait = min(1.0, deficit / self._rate)
            if stop is not None and stop.wait(wait):
                return False
            if stop is None:
                time.sleep(wait)


class Governor:
    """Watches 1-minute load average and scales both pools down when the machine is busy.

    Hysteresis is deliberate: throttle at 0.75 x nproc, restore only below 0.40 x nproc. A single
    threshold would oscillate every time somebody's job breathed, and each oscillation resizes a
    thread pool.

    The decision and the load that produced it are exposed via `status()` so the dashboard can show
    why concurrency dropped instead of leaving it a mystery.
    """

    NORMAL, THROTTLED, PAUSED = "normal", "throttled", "paused"

    def __init__(self, state, interval=None):
        self.state = state
        self.interval = interval or config.GOVERNOR_INTERVAL
        self.mode = self.NORMAL
        self.load = 0.0
        self._stop = threading.Event()
        self._thread = None

    @staticmethod
    def load_average():
        try:
            with open("/proc/loadavg") as handle:
                return float(handle.read().split()[0])
        except (OSError, ValueError, IndexError):
            return 0.0

    def factor(self):
        """Multiplier applied to every configured pool size."""
        return {self.NORMAL: 1.0, self.THROTTLED: 0.5, self.PAUSED: 0.0}[self.mode]

    def should_fetch(self):
        return self.mode != self.PAUSED

    def scale(self, n):
        return max(1, int(n * self.factor())) if self.factor() > 0 else 0

    def _tick(self):
        # The throughput table gets a row per completed image -- 360,000 of them over the run --
        # and only the last few minutes are ever read. Pruning here keeps it bounded without
        # needing a second background thread.
        try:
            self.state.prune_throughput()
        except Exception:  # noqa: BLE001 -- housekeeping must never stop the governor
            pass

        self.load = self.load_average()
        previous = self.mode

        if self.load > config.LOAD_PAUSE:
            self.mode = self.PAUSED
        elif self.load > config.LOAD_THROTTLE:
            self.mode = self.THROTTLED
        elif self.load < config.LOAD_RESTORE:
            self.mode = self.NORMAL
        # Between LOAD_RESTORE and LOAD_THROTTLE the mode is left alone -- that gap is the hysteresis.

        if self.mode != previous:
            self.state.log(
                "warn" if self.mode != self.NORMAL else "info",
                f"load {self.load:.1f} on {config.NPROC} cores -> {self.mode}"
                f" (thresholds: throttle {config.LOAD_THROTTLE:.0f}, pause {config.LOAD_PAUSE:.0f})",
                source="governor",
            )

    def run(self):
        while not self._stop.is_set():
            self._tick()
            self._stop.wait(self.interval)

    def start(self):
        self._tick()  # so the first status() call is not a lie
        self._thread = threading.Thread(target=self.run, name="governor", daemon=True)
        self._thread.start()
        return self

    def stop(self):
        self._stop.set()

    def status(self):
        return {
            "mode": self.mode,
            "load": round(self.load, 2),
            "nproc": config.NPROC,
            "factor": self.factor(),
            "throttle_at": round(config.LOAD_THROTTLE, 1),
            "pause_at": round(config.LOAD_PAUSE, 1),
        }
