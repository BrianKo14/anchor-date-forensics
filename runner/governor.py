"""Decides when to stand down. Replaces the downloader's bandwidth governor with a GPU one.

sapucay is shared -- 48 cores, two 3080s, eighteen other home directories -- and lightly used
rather than dedicated. A run that sits on both cards all night is the kind of thing that makes
the next person stop lending you the machine. The rules:

  * A GPU running someone else's compute process is theirs. We stop claiming work for that card
    and wait, rather than sharing it: a 10 GB card fits one serious job, and two jobs on it means
    both thrash and neither finishes sooner.
  * Load average near the core count means the box is busy with something other than us.
  * A PAUSE file anyone can create stops everything, because the dashboard binds to localhost and
    a colleague at 3am has no other way to ask.

Yielding is cheap here in a way it was not for the downloader: chunks are small and resumable, so
standing down costs the tail of one chunk, not a re-download.
"""

import getpass
import os
import shutil
import subprocess
import time

import config


def gpu_processes():
    """{gpu_index: [(pid, username)]} for every compute process nvidia-smi reports.

    nvidia-smi's per-process query does not include the GPU index in older drivers, so the
    association comes from querying each card's apps separately.
    """
    found = {}
    for index in config.GPUS:
        try:
            out = subprocess.check_output(
                ["nvidia-smi", "--id=" + str(index), "--query-compute-apps=pid",
                 "--format=csv,noheader"],
                text=True, stderr=subprocess.DEVNULL, timeout=15)
        except (subprocess.SubprocessError, FileNotFoundError, OSError):
            found[index] = []
            continue
        pids = [line.strip() for line in out.splitlines() if line.strip()]
        found[index] = [(pid, owner_of(pid)) for pid in pids]
    return found


def owner_of(pid):
    try:
        out = subprocess.check_output(["ps", "-o", "user=", "-p", str(pid)],
                                      text=True, stderr=subprocess.DEVNULL, timeout=10)
        return out.strip() or "?"
    except (subprocess.SubprocessError, OSError):
        return "?"


def gpu_memory():
    """{gpu_index: (used_mib, total_mib, util_pct)} for the dashboard."""
    try:
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=index,memory.used,memory.total,utilization.gpu",
             "--format=csv,noheader,nounits"], text=True, stderr=subprocess.DEVNULL, timeout=15)
    except (subprocess.SubprocessError, FileNotFoundError, OSError):
        return {}
    stats = {}
    for line in out.splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) == 4:
            stats[int(parts[0])] = (int(parts[1]), int(parts[2]), int(parts[3]))
    return stats


class Governor:
    """Polls the machine and answers 'may I use GPU n right now?'.

    Polled on an interval rather than checked per chunk: nvidia-smi costs ~100 ms and a chunk is
    minutes long, so a stale answer of up to GOVERNOR_INTERVAL is free, while shelling out from
    two workers on every claim is not.
    """

    def __init__(self, state):
        self.state = state
        self.me = getpass.getuser()
        self._blocked = {}          # gpu -> reason, or absent when free
        self._load = 0.0
        self._disk = {}
        self._checked = 0.0
        self._announced = {}

    def refresh(self, force=False):
        now = time.time()
        if not force and now - self._checked < config.GOVERNOR_INTERVAL:
            return
        self._checked = now

        self._load = os.getloadavg()[0]
        blocked = {}

        if self._load > config.LOAD_PAUSE:
            for gpu in config.GPUS:
                blocked[gpu] = f"load {self._load:.1f} > {config.LOAD_PAUSE:.0f}"

        # Checked on both volumes, not just the one we write results to: the failure this exists
        # to prevent was a cache landing somewhere nobody had thought about, on the volume shared
        # with everyone else's home directory.
        self._disk = {}
        for label, path in (("work", config.WORK_DIR), ("root", config.PROJECT_ROOT)):
            try:
                free = shutil.disk_usage(path).free
            except OSError:
                continue
            self._disk[label] = free
            if free < config.MIN_FREE_BYTES:
                for gpu in config.GPUS:
                    blocked[gpu] = (f"{label} volume has {free / 1e9:.1f} GB free, "
                                    f"below the {config.MIN_FREE_BYTES / 1e9:.0f} GB floor")

        if config.YIELD_TO_OTHER_GPU_PROCS:
            for gpu, procs in gpu_processes().items():
                others = sorted({owner for _, owner in procs if owner not in (self.me, "?")})
                if others:
                    blocked[gpu] = f"in use by {', '.join(others)}"

        # Log only transitions. A 12-hour run polling every 20 s would otherwise write 2,000
        # identical "still blocked" lines over the one line that says why it stalled.
        for gpu in config.GPUS:
            was, now_reason = self._announced.get(gpu), blocked.get(gpu)
            if was != now_reason:
                self._announced[gpu] = now_reason
                if now_reason:
                    self.state.log("warn", f"gpu {gpu}: standing down -- {now_reason}", "governor")
                else:
                    self.state.log("info", f"gpu {gpu}: free again, resuming", "governor")
        self._blocked = blocked

    def blocked(self, gpu):
        self.refresh()
        return self._blocked.get(gpu)

    def status(self):
        return {
            "load": round(self._load, 2),
            "load_pause": round(config.LOAD_PAUSE, 1),
            "blocked": {str(g): r for g, r in self._blocked.items()},
            "disk_free_gb": {k: round(v / 1e9, 1) for k, v in self._disk.items()},
            "disk_floor_gb": round(config.MIN_FREE_BYTES / 1e9),
            "memory": {str(g): m for g, m in gpu_memory().items()},
        }
