"""Orchestrator for the full AI-GenBench download. Run this; watch it in the browser.

    python -m downloader.pipeline --smoke --limit 20      # ~3 min end-to-end rehearsal
    python -m downloader.pipeline                         # the real ~30 h run

The four sources run in parallel threads because they contend for nothing: different hosts,
different bottlenecks. RAISE is ~29 h of the ~30 h total, so everything else finishes inside it and
the wall clock is set by the slowest source rather than by their sum.

Interrupting is safe at any point. Images are written to `.part` and renamed, shards and zips resume
from a byte offset, and anything left `active` in the database is returned to `pending` on the next
start. The reconcile pass makes the filesystem authoritative, so even losing the database entirely
costs a rescan rather than a re-download.
"""

import argparse
import signal
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import config  # noqa: E402  -- must come first; it sets AIGENBENCH_DATA_ROOT before common.py reads it
import context  # noqa: E402
import ratelimit  # noqa: E402
import state as state_module  # noqa: E402
from sources import coco, fakes_hf, laion, raise_tiff  # noqa: E402

MODULES = {
    "raise": raise_tiff,
    "fakes": fakes_hf,
    "coco": coco,
    "laion": laion,
}


def plan_all(ctx, sources):
    """Populate the work queues. Idempotent -- re-planning never disturbs completed rows."""
    print("\n--- planning ---")
    for name in sources:
        total = MODULES[name].plan(ctx)
        ctx.state.set_control(f"total:{name}", total)
    print()


def run_all(ctx, sources):
    """One thread per source; returns when all have finished or the run is stopping."""
    threads = []
    results = {}

    def worker(name):
        started = time.time()
        try:
            done, failed = MODULES[name].run(ctx)
            results[name] = (done, failed)
            ctx.log("info", f"finished: {done} done, {failed} failed, "
                            f"{(time.time() - started) / 60:.1f} min", name)
        except SystemExit as error:
            ctx.log("error", f"stopped: {error}", name)
            results[name] = (0, 0)
        except Exception as error:  # noqa: BLE001 -- one source must not take the others down
            ctx.log("error", f"crashed: {type(error).__name__}: {error}", name)
            results[name] = (0, 0)

    for name in sources:
        thread = threading.Thread(target=worker, args=(name,), name=f"src-{name}", daemon=True)
        thread.start()
        threads.append(thread)
        # Stagger slightly so four sources do not all resolve DNS and open sockets in the same
        # millisecond; it makes the first seconds of the throughput graph readable.
        time.sleep(0.3)

    for thread in threads:
        while thread.is_alive():
            thread.join(timeout=1.0)
    return results


def serve(ctx):
    """Start the dashboard in a background thread. Failure here must not stop the download."""
    try:
        import uvicorn

        import server as server_module
    except ImportError as error:
        ctx.log("warn", f"dashboard unavailable ({error}); downloading without it", "server")
        return None

    app = server_module.build_app(ctx)
    settings = uvicorn.Config(app, host=config.HOST, port=config.PORT, log_level="warning")
    instance = uvicorn.Server(settings)
    thread = threading.Thread(target=instance.run, name="dashboard", daemon=True)
    thread.start()
    ctx.log("info", f"dashboard on http://{config.HOST}:{config.PORT}"
                    f"  (ssh -F ssh_config -L {config.PORT}:127.0.0.1:{config.PORT} remote)",
            "server")
    return instance


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sources", nargs="+", choices=list(MODULES), default=list(config.SOURCE_ORDER),
                        help="subset of sources to run (default: all, RAISE first)")
    parser.add_argument("--smoke", action="store_true",
                        help="rehearse the whole path on a few images per source")
    parser.add_argument("--limit", type=int, default=20,
                        help="images per source in --smoke mode (default 20)")
    parser.add_argument("--plan-only", action="store_true",
                        help="populate the queues and exit without downloading")
    parser.add_argument("--reconcile", action="store_true",
                        help="rebuild item status from what is on disk, then exit")
    parser.add_argument("--retry-failed", action="store_true",
                        help="return failed items to the queue before starting")
    parser.add_argument("--no-server", action="store_true", help="skip the dashboard")
    parser.add_argument("--no-manifest", action="store_true",
                        help="skip manifest generation at the end")
    arguments = parser.parse_args()

    config.ensure_dirs()
    for path in config.install_notes():
        print(f"wrote {path}")
    print(config.describe())

    db = state_module.State()

    if arguments.reconcile:
        state_module.reconcile(db)
        return

    requeued = db.requeue_active()
    if requeued:
        print(f"requeued {requeued} items left active by a previous run")
    if arguments.retry_failed:
        print(f"requeued {db.retry_failed()} previously failed items")

    governor = ratelimit.Governor(db).start()
    ctx = context.Context(db, governor=governor, smoke=arguments.smoke,
                          limit=arguments.limit if arguments.smoke else None)

    def handle_signal(signum, _frame):
        print(f"\nsignal {signum} -- finishing in-flight work, then stopping", flush=True)
        ctx.stop.set()

    signal.signal(signal.SIGINT, handle_signal)
    signal.signal(signal.SIGTERM, handle_signal)

    sources = [s for s in config.SOURCE_ORDER if s in arguments.sources]
    plan_all(ctx, sources)
    if arguments.plan_only:
        print(db.snapshot_json())
        return

    if not arguments.no_server:
        serve(ctx)

    print(f"--- running {', '.join(sources)} ---\n")
    started = time.time()
    run_all(ctx, sources)
    governor.stop()

    print(f"\n--- done in {(time.time() - started) / 60:.1f} min ---")
    for source, counts in sorted(db.counts().items()):
        print(f"{source:<8} {counts}")

    if not arguments.no_manifest and not ctx.stop.is_set():
        import manifest
        manifest.build(ctx)

    db.close()


if __name__ == "__main__":
    main()
