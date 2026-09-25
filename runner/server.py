"""Read-only status feed for the scoring run, plus the few controls that change a live one.

Bound to 127.0.0.1, never 0.0.0.0: sapucay carries eighteen other home directories and this
endpoint can pause a running job. Reach it with an SSH tunnel:

    ssh -F ssh_config -L 8766:127.0.0.1:8766 remote

Controls are written to the `control` table rather than into worker memory, because the workers
re-read that table between chunks. That indirection is what lets a pause outlive a dashboard
restart.
"""

import shutil
import time

import config


def eta_seconds(state, detector, remaining_images):
    per_image = state.per_image_seconds(detector)
    if not per_image or remaining_images <= 0:
        return None
    # Divided across however many GPUs are actually working, since per_image is measured per
    # worker; with both cards free the queue drains at roughly twice one worker's rate.
    workers = max(1, len(config.GPUS))
    return round(remaining_images * per_image / workers)


def snapshot(state, gov, stop, started):
    counts = state.counts()
    control = state.all_control()
    rates = state.recent_rate()

    detectors = []
    for name in config.PANEL_ORDER:
        entry = counts.get(name)
        if not entry:
            continue
        done = entry.get("images_done", 0)
        total = entry.get("images_total", 0)
        per_image = state.per_image_seconds(name)
        detectors.append({
            "name": name,
            "images_done": done,
            "images_total": total,
            "pct": round(100 * done / total, 1) if total else 0.0,
            "chunks": {k: entry.get(k, 0) for k in ("pending", "active", "done", "failed")},
            "ms_per_image": round(per_image * 1000) if per_image else None,
            "images_per_s": round(rates.get(name, 0.0), 1),
            "eta_seconds": eta_seconds(state, name, total - done),
            "paused": state.is_paused(name),
        })

    usage = shutil.disk_usage(config.WORK_DIR)
    images_done = sum(d["images_done"] for d in detectors)
    images_total = sum(d["images_total"] for d in detectors)
    return {
        "elapsed": round(time.time() - started),
        "stopping": stop.is_set(),
        "paused": control.get("paused") == "1",
        "courtesy_paused": config.PAUSE_FILE.exists(),
        "pause_file": str(config.PAUSE_FILE),
        "governor": gov.status(),
        "disk": {"free": usage.free, "total": usage.total,
                 "used_pct": round(100 * usage.used / usage.total, 1)},
        "overall": {
            "images_done": images_done,
            "images_total": images_total,
            "pct": round(100 * images_done / images_total, 1) if images_total else 0.0,
            "images_per_s": round(rates.get("total", 0.0), 1),
        },
        "detectors": detectors,
        "failures": state.failures(),
        "events": state.recent_events(30),
    }


def build_app(state, gov, stop, started):
    from fastapi import FastAPI
    from fastapi.responses import FileResponse, JSONResponse

    app = FastAPI(title="panel scoring", docs_url=None, redoc_url=None)
    index = config.PROJECT_ROOT / "runner" / "static" / "index.html"

    @app.get("/")
    def root():
        return FileResponse(index)

    @app.get("/api/status")
    def status():
        return JSONResponse(snapshot(state, gov, stop, started))

    @app.post("/api/control")
    async def control(payload: dict):
        allowed = {"paused"} | {f"paused:{name}" for name in config.PANEL_ORDER}
        applied = {}
        for key, value in payload.items():
            if key not in allowed:
                return JSONResponse({"error": f"unknown control {key}"}, status_code=400)
            state.set_control(key, value)
            applied[key] = str(value)
        state.log("info", f"control: {applied}", "dashboard")
        return {"ok": True, "applied": applied}

    @app.post("/api/action")
    async def action(payload: dict):
        what = payload.get("action")
        if what == "retry-failed":
            n = state.retry_failed(payload.get("detector"))
            state.log("info", f"requeued {n} failed chunks", "dashboard")
            return {"ok": True, "requeued": n}
        if what == "stop":
            stop.set()
            state.log("warn", "stop requested from the dashboard", "dashboard")
            return {"ok": True}
        return JSONResponse({"error": f"unknown action {what}"}, status_code=400)

    return app
