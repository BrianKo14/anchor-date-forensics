"""The dashboard: a read-only status feed plus the handful of controls that change a live run.

Bound to 127.0.0.1 only, never 0.0.0.0. sapucay has eighteen other people's home directories on it
and this endpoint can pause a colleague's... no, only ours -- but it can also unthrottle a download
onto a shared uplink, which is reason enough not to publish it on the LAN. Reach it with an SSH
tunnel:

    ssh -F ssh_config -L 8765:127.0.0.1:8765 remote

Controls are written to the `control` table rather than into the workers' memory, because the
workers re-read that table every batch. That indirection is what lets a pause survive the dashboard
being restarted, and what would let a second process drive the run if it ever needed to.
"""

import shutil
import time
from collections import deque

import config

# Rolling (timestamp, done-count) history per source, used for the completion-rate ETA. Kept in
# memory rather than in the database: it is derived, cheap to rebuild, and writing a row per poll
# would add 1,800 pointless inserts an hour.
_history = {}
_HISTORY = 30


def _rate_and_eta(source, done, total):
    now = time.time()
    series = _history.setdefault(source, deque(maxlen=_HISTORY))
    series.append((now, done))
    if len(series) < 2:
        return 0.0, None
    (then, before), (now, after) = series[0], series[-1]
    span = now - then
    if span <= 0 or after <= before:
        return 0.0, None
    rate = (after - before) / span
    remaining = max(0, (total or 0) - done)
    return rate, (remaining / rate if rate > 0 else None)


def snapshot(ctx):
    state = ctx.state
    counts = state.counts()
    control = state.all_control()
    rates = state.recent_rate()

    sources = []
    for name in config.SOURCE_ORDER:
        entry = counts.get(name, {})
        done = entry.get("done", 0)
        try:
            total = int(control.get(f"total:{name}", 0))
        except (TypeError, ValueError):
            total = 0
        items_per_s, eta = _rate_and_eta(name, done, total)
        sources.append({
            "name": name,
            "total": total,
            "done": done,
            "failed": entry.get("failed", 0),
            "pending": entry.get("pending", 0),
            "active": entry.get("active", 0),
            "bytes": entry.get("bytes", 0),
            "bytes_per_s": round(rates.get(name, 0.0)),
            "items_per_s": round(items_per_s, 2),
            "eta_seconds": round(eta) if eta else None,
            "concurrency": state.concurrency(name),
            "effective_concurrency": ctx.concurrency(name),
            "hard_cap": config.HARD_CONCURRENCY_CAP.get(name),
            "paused": state.is_paused(name),
            "failures": state.failure_reasons(name),
        })

    usage = shutil.disk_usage(config.DATA_ROOT)
    return {
        "elapsed": round(ctx.elapsed),
        "stopping": ctx.stop.is_set(),
        "governor": ctx.governor.status(),
        "disk": {
            "free": usage.free,
            "total": usage.total,
            "used_pct": round(100 * usage.used / usage.total, 1),
        },
        "courtesy_paused": ctx.courtesy_paused(),
        "pause_file": str(config.PAUSE_FILE),
        "rate_bytes": state.rate_bytes(),
        "rate_default": config.DEFAULT_RATE_BYTES,
        "cpu_workers": int(control.get("cpu_workers", config.DEFAULT_CPU_WORKERS)),
        "paused": control.get("paused") == "1",
        "total_bytes_per_s": round(rates.get("total", 0.0)),
        "sources": sources,
        "artifacts": [
            {"name": a["name"], "status": a["status"], "got": a["got_bytes"],
             "expected": a["expected_bytes"], "error": a["error"]}
            for a in state.artifacts()
            if a["status"] != "done" or a["source"] == "coco"
        ][:20],
        "events": state.recent_events(30),
        "data_root": str(config.DATA_ROOT),
    }


def build_app(ctx):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="AI-GenBench download", docs_url=None, redoc_url=None)
    index = config.PROJECT_ROOT / "downloader" / "static" / "index.html"

    @app.get("/")
    def root():
        return FileResponse(index)

    @app.get("/api/status")
    def status():
        return JSONResponse(snapshot(ctx))

    @app.post("/api/control")
    async def control(payload: dict):
        """Set control values. Anything unrecognised is rejected rather than silently stored."""
        allowed = (
            {"paused", "rate_bytes", "cpu_workers"}
            | {f"paused:{s}" for s in config.SOURCES}
            | {f"concurrency:{s}" for s in config.SOURCES}
        )
        applied = {}
        for key, value in payload.items():
            if key not in allowed:
                return JSONResponse({"error": f"unknown control {key}"}, status_code=400)
            ctx.state.set_control(key, value)
            applied[key] = str(value)
        # Take the rate change immediately rather than waiting for a worker to notice.
        ctx.sync_rate()
        ctx.state.log("info", f"control: {applied}", source="dashboard")
        return {"ok": True, "applied": applied}

    @app.post("/api/action")
    async def action(payload: dict):
        what = payload.get("action")
        if what == "retry-failed":
            n = ctx.state.retry_failed(payload.get("source"))
            ctx.state.log("info", f"requeued {n} failed items", source="dashboard")
            return {"ok": True, "requeued": n}
        if what == "stop":
            ctx.stop.set()
            ctx.state.log("warn", "stop requested from the dashboard", source="dashboard")
            return {"ok": True}
        return JSONResponse({"error": f"unknown action {what}"}, status_code=400)

    return app
