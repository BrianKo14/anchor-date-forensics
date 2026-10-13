"""Several self-description runs back to back, unattended. Normally started through
`./runner/run.sh selfdesc queue <queue-file>` (or `queue-smoke`).

A queue file has one run per line: `<manifest> <out-dir> [runner.selfdesc options...]`, shell
quoting allowed, `#` comments and blank lines ignored. Relative paths are relative to the queue
file's directory. Runs go in order; the queue stops at the first run that exits non-zero, so a
broken run is never followed by one that depends on it. Every run is itself resumable, so
starting the same queue again skips what is finished (prepare, reconcile and the temporal cache
make a finished run take seconds) and continues where the stop happened.

With --smoke every line runs as `--smoke` (each run then writes to <out-dir>-smoke).

SIGTERM / SIGINT are passed on to the run in progress, which finishes its shards in flight and
exits; the queue then stops without starting the next run.
"""

import argparse
import shlex
import signal
import subprocess
import sys
import time
from pathlib import Path


def parse(queue_file):
    base = queue_file.resolve().parent
    runs = []
    for number, raw in enumerate(queue_file.read_text().splitlines(), 1):
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        words = shlex.split(line)
        if len(words) < 2:
            sys.exit(f"{queue_file}:{number}: need '<manifest> <out-dir> [options]'")
        manifest, out_dir = (str((base / w).resolve()) if not w.startswith("/") else w
                             for w in words[:2])
        if not Path(manifest).is_file():
            sys.exit(f"{queue_file}:{number}: no manifest at {manifest}")
        runs.append((number, manifest, out_dir, words[2:]))
    if not runs:
        sys.exit(f"{queue_file}: no runs")
    return runs


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("queue", type=Path)
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    runs = parse(args.queue)

    child = {"process": None}
    stopping = {"flag": False}

    def handle(signum, _frame):
        stopping["flag"] = True
        print(f"\nqueue: signal {signum}, stopping the current run and not starting another",
              flush=True)
        if child["process"] is not None and child["process"].poll() is None:
            child["process"].send_signal(signal.SIGTERM)

    signal.signal(signal.SIGINT, handle)
    signal.signal(signal.SIGTERM, handle)

    started = time.time()
    for i, (number, manifest, out_dir, extra) in enumerate(runs, 1):
        command = [sys.executable, "-m", "runner.selfdesc", "--manifest", manifest,
                   "--out-dir", out_dir, *extra, *(["--smoke"] if args.smoke else [])]
        print(f"\n=== queue {i}/{len(runs)} (line {number}): {shlex.join(command[3:])} ===\n",
              flush=True)
        t = time.time()
        child["process"] = subprocess.Popen(command)
        code = child["process"].wait()
        print(f"\n=== queue {i}/{len(runs)}: exit {code} after {(time.time() - t) / 60:.1f} min "
              f"===", flush=True)
        if code != 0 or stopping["flag"]:
            print(f"queue stopped at run {i}/{len(runs)}; start the same queue again to resume")
            sys.exit(code or 1)
    print(f"\nqueue finished: {len(runs)} run(s) in {(time.time() - started) / 3600:.1f} h")


if __name__ == "__main__":
    main()
