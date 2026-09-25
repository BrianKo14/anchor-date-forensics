"""SQLite job state for the scoring run: which chunks are done, and what the operator wants.

Adapted from the downloader branch's state.py. Same two ideas, because both earned their keep
over a 30-hour unattended run:

  * The database is an *accelerator*, not the source of truth. A chunk's score CSV is written to
    `.part` and renamed, so a file that exists is complete by construction, and `reconcile()` can
    rebuild every status by walking the output directory. Losing the database costs a rescan,
    never re-scoring.

  * Controls live in a table the workers re-read each chunk, not in their memory. That is what
    lets a pause survive a dashboard restart, and what lets a second process drive the run.

What changed: the work unit is a chunk of manifest rows scored by one detector, not an image to
download, so there is no bytes/rate-limit machinery and the throughput number is images per
second derived from recent completions rather than a separate table.
"""

import json
import sqlite3
import threading
import time

import config

PENDING, ACTIVE, DONE, FAILED = "pending", "active", "done", "failed"

SCHEMA = """
CREATE TABLE IF NOT EXISTS chunks (
    id          INTEGER PRIMARY KEY,
    detector    TEXT NOT NULL,
    seq         INTEGER NOT NULL,
    start       INTEGER NOT NULL,
    n           INTEGER NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    gpu         INTEGER,
    elapsed_s   REAL,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_chunks_key ON chunks(detector, seq);
CREATE INDEX IF NOT EXISTS idx_chunks_claim ON chunks(status, id);
CREATE INDEX IF NOT EXISTS idx_chunks_done ON chunks(status, updated_at);

CREATE TABLE IF NOT EXISTS control (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS events (
    ts      REAL NOT NULL,
    source  TEXT,
    level   TEXT NOT NULL,
    msg     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);
"""

DEFAULT_CONTROL = {
    "paused": "0",
    **{f"paused:{name}": "0" for name in config.PANEL_ORDER},
}


class State:
    """Thread- and process-safe wrapper over one SQLite connection.

    WAL plus a 30 s busy timeout, because unlike the downloader this database genuinely has
    several *processes* on it: the orchestrator, and whatever the dashboard is doing. Write
    volume is a row per finished chunk -- tens per hour -- so contention is theoretical.
    """

    def __init__(self, path=None):
        self.path = path or config.STATE_DB
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._db = sqlite3.connect(str(self.path), check_same_thread=False, timeout=30)
        self._db.row_factory = sqlite3.Row
        self._db.executescript(SCHEMA)
        self._db.execute("PRAGMA journal_mode=WAL")
        self._db.execute("PRAGMA synchronous=NORMAL")
        self._db.commit()
        self._init_control()

    # -- control ---------------------------------------------------------------------------------

    def _init_control(self):
        with self._lock:
            for key, value in DEFAULT_CONTROL.items():
                self._db.execute("INSERT OR IGNORE INTO control (k, v) VALUES (?, ?)", (key, value))
            self._db.commit()

    def get_control(self, key, default=None):
        with self._lock:
            row = self._db.execute("SELECT v FROM control WHERE k = ?", (key,)).fetchone()
        return row["v"] if row else default

    def set_control(self, key, value):
        with self._lock:
            self._db.execute(
                "INSERT INTO control (k, v) VALUES (?, ?) ON CONFLICT(k) DO UPDATE SET v = excluded.v",
                (key, str(value)),
            )
            self._db.commit()

    def all_control(self):
        with self._lock:
            rows = self._db.execute("SELECT k, v FROM control").fetchall()
        return {row["k"]: row["v"] for row in rows}

    def is_paused(self, detector=None):
        if self.get_control("paused", "0") == "1":
            return True
        return detector is not None and self.get_control(f"paused:{detector}", "0") == "1"

    # -- chunks ----------------------------------------------------------------------------------

    def add_chunks(self, rows):
        """Bulk-insert the plan. Re-planning leaves completed chunks alone -- unless their geometry
        moved.

        A plain INSERT OR IGNORE is wrong here. Chunks are keyed by (detector, seq), but `seq` only
        means anything relative to a chunk size: re-planning the same manifest at 500 rows per chunk
        instead of 20 makes seq 0 a different span of the manifest while the key stays identical.
        The old row survives, reconcile() finds the old 20-row output still on disk, marks the chunk
        done, and the merge silently ships 20 scores where 500 were planned. So a changed (start, n)
        resets the chunk to pending.
        """
        with self._lock:
            self._db.executemany(
                "INSERT INTO chunks (detector, seq, start, n, status, updated_at)"
                " VALUES (?, ?, ?, ?, 'pending', ?)"
                " ON CONFLICT(detector, seq) DO UPDATE SET"
                "   status = CASE WHEN chunks.start != excluded.start OR chunks.n != excluded.n"
                "                 THEN 'pending' ELSE chunks.status END,"
                "   error = CASE WHEN chunks.start != excluded.start OR chunks.n != excluded.n"
                "                THEN NULL ELSE chunks.error END,"
                "   attempts = CASE WHEN chunks.start != excluded.start OR chunks.n != excluded.n"
                "                   THEN 0 ELSE chunks.attempts END,"
                "   start = excluded.start,"
                "   n = excluded.n,"
                "   updated_at = excluded.updated_at",
                [(r["detector"], r["seq"], r["start"], r["n"], time.time()) for r in rows],
            )
            self._db.commit()

    def claim(self, detectors=None):
        """Atomically take the next pending chunk and mark it active. None when there is none.

        Ordered by id, which is plan order, which is config.PANEL_ORDER -- cheapest detector
        first. Two GPU workers sharing this call is the whole reason it is one transaction.
        """
        with self._lock:
            if detectors:
                placeholders = ",".join("?" * len(detectors))
                row = self._db.execute(
                    f"SELECT * FROM chunks WHERE status = 'pending'"
                    f" AND detector IN ({placeholders}) ORDER BY id LIMIT 1",
                    tuple(detectors),
                ).fetchone()
            else:
                row = self._db.execute(
                    "SELECT * FROM chunks WHERE status = 'pending' ORDER BY id LIMIT 1").fetchone()
            if row is None:
                return None
            self._db.execute("UPDATE chunks SET status = 'active', updated_at = ? WHERE id = ?",
                             (time.time(), row["id"]))
            self._db.commit()
            return dict(row)

    def finish(self, chunk_id, status, elapsed_s=None, gpu=None, error=None):
        with self._lock:
            self._db.execute(
                "UPDATE chunks SET status = ?, elapsed_s = ?, gpu = ?, error = ?,"
                " attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (status, elapsed_s, gpu, error, time.time(), chunk_id),
            )
            self._db.commit()

    def release(self, chunk_id, error=None):
        """Hand an active chunk back to pending -- a worker interrupted mid-flight."""
        with self._lock:
            self._db.execute(
                "UPDATE chunks SET status = 'pending', error = ?, updated_at = ? WHERE id = ?",
                (error, time.time(), chunk_id))
            self._db.commit()

    def requeue_active(self):
        """Startup repair: nothing can legitimately be active before the first worker starts."""
        with self._lock:
            cursor = self._db.execute("UPDATE chunks SET status = 'pending' WHERE status = 'active'")
            self._db.commit()
            return cursor.rowcount

    def retry_failed(self, detector=None):
        with self._lock:
            if detector:
                cursor = self._db.execute(
                    "UPDATE chunks SET status = 'pending', error = NULL, attempts = 0"
                    " WHERE status = 'failed' AND detector = ?", (detector,))
            else:
                cursor = self._db.execute(
                    "UPDATE chunks SET status = 'pending', error = NULL, attempts = 0"
                    " WHERE status = 'failed'")
            self._db.commit()
            return cursor.rowcount

    def exhausted(self, chunk_id):
        """True when a chunk has burned through MAX_ATTEMPTS and should stop being retried."""
        with self._lock:
            row = self._db.execute("SELECT attempts FROM chunks WHERE id = ?", (chunk_id,)).fetchone()
        return bool(row) and row["attempts"] >= config.MAX_ATTEMPTS

    def counts(self):
        """{detector: {status: chunks, 'images_done': n, 'images_total': n}}."""
        with self._lock:
            rows = self._db.execute(
                "SELECT detector, status, COUNT(*) AS chunks, COALESCE(SUM(n), 0) AS images"
                " FROM chunks GROUP BY detector, status").fetchall()
        out = {}
        for row in rows:
            entry = out.setdefault(row["detector"], {"images_done": 0, "images_total": 0})
            entry[row["status"]] = row["chunks"]
            entry["images_total"] += row["images"]
            if row["status"] == DONE:
                entry["images_done"] = row["images"]
        return out

    def recent_rate(self, window=600.0):
        """{detector: images/s} from chunks finished inside `window`, plus a 'total'.

        Derived from completions rather than sampled, so a detector that finishes one 2,500-image
        chunk every nine minutes still reports a sane number instead of oscillating between zero
        and a spike.
        """
        cutoff = time.time() - window
        with self._lock:
            rows = self._db.execute(
                "SELECT detector, COALESCE(SUM(n), 0) AS images, COALESCE(SUM(elapsed_s), 0) AS secs"
                " FROM chunks WHERE status = 'done' AND updated_at > ? GROUP BY detector",
                (cutoff,)).fetchall()
        rates = {r["detector"]: (r["images"] / r["secs"] if r["secs"] else 0.0) for r in rows}
        rates["total"] = sum(rates.values())
        return rates

    def per_image_seconds(self, detector):
        """Mean seconds per image over every completed chunk. None until one finishes."""
        with self._lock:
            row = self._db.execute(
                "SELECT COALESCE(SUM(n), 0) AS images, COALESCE(SUM(elapsed_s), 0) AS secs"
                " FROM chunks WHERE detector = ? AND status = 'done'", (detector,)).fetchone()
        return (row["secs"] / row["images"]) if row["images"] else None

    def pending_exists(self):
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM chunks WHERE status IN ('pending', 'active') LIMIT 1").fetchone()
        return row is not None

    def chunks_for(self, detector, status=DONE):
        with self._lock:
            return [dict(r) for r in self._db.execute(
                "SELECT * FROM chunks WHERE detector = ? AND status = ? ORDER BY seq",
                (detector, status)).fetchall()]

    def failures(self, limit=12):
        with self._lock:
            rows = self._db.execute(
                "SELECT detector, seq, attempts, error FROM chunks WHERE status = 'failed'"
                " ORDER BY updated_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    # -- telemetry -------------------------------------------------------------------------------

    def log(self, level, msg, source=None):
        with self._lock:
            self._db.execute("INSERT INTO events (ts, source, level, msg) VALUES (?, ?, ?, ?)",
                             (time.time(), source, level, msg))
            self._db.commit()
        print(f"[{level}] {source or '-':<12} {msg}", flush=True)

    def recent_events(self, limit=40):
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def snapshot_json(self):
        return json.dumps({"counts": self.counts(), "control": self.all_control()}, indent=2)

    def close(self):
        with self._lock:
            self._db.close()


def chunk_score_path(detector, seq):
    return config.WORK_DIR / "scores" / detector / f"{seq:05d}.csv"


def scored_rows(path):
    """Data rows in a chunk's score CSV, or -1 if it cannot be read."""
    try:
        with path.open() as handle:
            return max(0, sum(1 for _ in handle) - 1)
    except OSError:
        return -1


def chunk_manifest_path(chunk_size, seq):
    """Keyed by chunk size, not detector: members sharing a size share the slice files."""
    return config.WORK_DIR / "chunks" / str(chunk_size) / f"{seq:05d}.csv"


def reconcile(state, verbose=True):
    """Repair chunk status from what is actually on disk.

    Ground truth is the output directory: a chunk's CSV is written to `.part` and renamed, so a
    file that exists is a complete chunk. This is what turns a lost database into a rescan rather
    than a re-score of everything that had already finished.
    """
    with state._lock:
        rows = state._db.execute("SELECT id, detector, seq, n, status FROM chunks").fetchall()

    to_pending, to_done = [], []
    for row in rows:
        path = chunk_score_path(row["detector"], row["seq"])
        # Complete means the right length, not merely present. An output left over from a plan at a
        # different chunk size sits at exactly this path with the wrong number of rows, and
        # accepting it would put a short score file into the merge.
        present = path.exists() and path.stat().st_size > 0 and scored_rows(path) == row["n"]
        if row["status"] == DONE and not present:
            to_pending.append(row["id"])
        elif row["status"] in (PENDING, ACTIVE, FAILED) and present:
            to_done.append(row["id"])

    with state._lock:
        if to_pending:
            state._db.executemany(
                "UPDATE chunks SET status = 'pending', updated_at = ? WHERE id = ?",
                [(time.time(), i) for i in to_pending])
        if to_done:
            state._db.executemany(
                "UPDATE chunks SET status = 'done', updated_at = ? WHERE id = ?",
                [(time.time(), i) for i in to_done])
        state._db.commit()

    if verbose:
        print(f"reconcile: {len(to_done)} found on disk marked done, "
              f"{len(to_pending)} missing outputs requeued")
    return len(to_done), len(to_pending)
