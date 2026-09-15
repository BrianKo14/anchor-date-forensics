"""SQLite job state: what has been downloaded, what is in flight, and what the operator wants.

Three things live here and they have different lifetimes:

  items      one row per image. 360,000 of them; the bulk of the database.
  artifacts  one row per large single file (3 COCO zips, 72 parquet shards) that is fetched with
             byte-range resume rather than all-or-nothing.
  control    the operator's live settings -- paused, rate limit, per-source concurrency. Workers
             re-read this every iteration, which is how the browser can pause a running job.

The database is an *accelerator*, not the source of truth. A file present on disk is complete by
construction (everything is written to `.part` and renamed), so `reconcile()` can rebuild the item
states by walking the tree. Losing the database costs a rescan, never a re-download.
"""

import json
import sqlite3
import threading
import time

import config

PENDING, ACTIVE, DONE, FAILED, SKIPPED = "pending", "active", "done", "failed", "skipped"

SCHEMA = """
CREATE TABLE IF NOT EXISTS items (
    id          INTEGER PRIMARY KEY,
    source      TEXT NOT NULL,
    split       TEXT NOT NULL,
    file_id     TEXT NOT NULL,
    url         TEXT,
    dest        TEXT NOT NULL,
    status      TEXT NOT NULL DEFAULT 'pending',
    bytes       INTEGER DEFAULT 0,
    sha256      TEXT,
    attempts    INTEGER NOT NULL DEFAULT 0,
    error       TEXT,
    updated_at  REAL
);
CREATE UNIQUE INDEX IF NOT EXISTS idx_items_key ON items(source, split, file_id);
CREATE INDEX IF NOT EXISTS idx_items_claim ON items(source, status, id);

CREATE TABLE IF NOT EXISTS artifacts (
    name            TEXT PRIMARY KEY,
    source          TEXT NOT NULL,
    url             TEXT NOT NULL,
    dest            TEXT NOT NULL,
    expected_bytes  INTEGER,
    got_bytes       INTEGER DEFAULT 0,
    md5             TEXT,
    sha256          TEXT,
    status          TEXT NOT NULL DEFAULT 'pending',
    error           TEXT,
    updated_at      REAL
);

CREATE TABLE IF NOT EXISTS control (k TEXT PRIMARY KEY, v TEXT NOT NULL);

CREATE TABLE IF NOT EXISTS events (
    ts      REAL NOT NULL,
    source  TEXT,
    level   TEXT NOT NULL,
    msg     TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_events_ts ON events(ts DESC);

CREATE TABLE IF NOT EXISTS throughput (
    ts      REAL NOT NULL,
    source  TEXT NOT NULL,
    bytes   INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_throughput_ts ON throughput(ts DESC);
"""

DEFAULT_CONTROL = {
    "paused": "0",
    "rate_bytes": str(config.DEFAULT_RATE_BYTES),
    "cpu_workers": str(config.DEFAULT_CPU_WORKERS),
    **{f"paused:{s}": "0" for s in config.SOURCES},
    **{f"concurrency:{s}": str(n) for s, n in config.DEFAULT_CONCURRENCY.items()},
}


class State:
    """Thread-safe wrapper over one SQLite connection.

    One connection guarded by a lock rather than a connection per thread: the write volume is tiny
    (a row update per completed image, so tens per second at most) and a single connection makes
    the claim-a-job transaction trivially correct. WAL keeps the dashboard's reads from ever
    blocking a worker's write.
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

    def is_paused(self, source=None):
        if self.get_control("paused", "0") == "1":
            return True
        return source is not None and self.get_control(f"paused:{source}", "0") == "1"

    def concurrency(self, source):
        """Configured concurrency for a source, clamped by the politeness cap.

        The cap is applied here rather than at the UI so it holds however the value was set --
        including by someone editing the database by hand.
        """
        try:
            want = int(self.get_control(f"concurrency:{source}", "4"))
        except (TypeError, ValueError):
            want = config.DEFAULT_CONCURRENCY.get(source, 4)
        want = max(1, min(want, config.MAX_CONCURRENCY))
        hard = config.HARD_CONCURRENCY_CAP.get(source)
        return min(want, hard) if hard else want

    def rate_bytes(self):
        try:
            return max(0, int(self.get_control("rate_bytes", config.DEFAULT_RATE_BYTES)))
        except (TypeError, ValueError):
            return config.DEFAULT_RATE_BYTES

    # -- items -----------------------------------------------------------------------------------

    def add_items(self, rows):
        """Bulk-insert planned work. Idempotent: re-planning never disturbs completed rows."""
        with self._lock:
            self._db.executemany(
                "INSERT OR IGNORE INTO items (source, split, file_id, url, dest, status, updated_at)"
                " VALUES (?, ?, ?, ?, ?, 'pending', ?)",
                [(r["source"], r["split"], r["file_id"], r.get("url"), r["dest"], time.time())
                 for r in rows],
            )
            self._db.commit()

    def add_done_items(self, rows):
        """Record already-written images in one transaction.

        The fake half does not go through the claim/fetch loop -- its images come out of parquet row
        groups in bulk -- but it still has to appear in `items`, or the dashboard reports it as 0 of
        180,000 forever and `reconcile` cannot see it. Inserted as done, with ON CONFLICT so a
        re-extraction updates rather than raises.
        """
        with self._lock:
            self._db.executemany(
                "INSERT INTO items (source, split, file_id, url, dest, status, bytes, updated_at)"
                " VALUES (?, ?, ?, NULL, ?, 'done', ?, ?)"
                " ON CONFLICT(source, split, file_id) DO UPDATE SET"
                " status = 'done', bytes = excluded.bytes, updated_at = excluded.updated_at",
                [(r["source"], r["split"], r["file_id"], r["dest"], r.get("bytes", 0), time.time())
                 for r in rows],
            )
            self._db.commit()

    def claim(self, source, limit=1, splits=None):
        """Atomically move up to `limit` pending rows to active and return them.

        Claiming inside one transaction is what lets several fetch threads share a source without
        two of them picking up the same image.

        `splits`, when given, restricts the claim to those split values. Without it, claiming pulls
        strictly by ascending id -- fine for a single-split source, but for a source whose splits
        were enqueued at very different oversample ratios (LAION: train 2.4x, one split's oversampled
        backlog has lower ids than the other's and will keep getting claimed long after that split has
        already met its target, starving the split that still needs work. Restricting to the splits
        that are still short makes claiming target-aware instead of purely id-ordered.
        """
        with self._lock:
            if splits:
                placeholders = ",".join("?" * len(splits))
                rows = self._db.execute(
                    f"SELECT * FROM items WHERE source = ? AND status = 'pending'"
                    f" AND split IN ({placeholders}) ORDER BY id LIMIT ?",
                    (source, *splits, limit),
                ).fetchall()
            else:
                rows = self._db.execute(
                    "SELECT * FROM items WHERE source = ? AND status = 'pending'"
                    " ORDER BY id LIMIT ?", (source, limit),
                ).fetchall()
            if rows:
                self._db.executemany(
                    "UPDATE items SET status = 'active', updated_at = ? WHERE id = ?",
                    [(time.time(), row["id"]) for row in rows],
                )
                self._db.commit()
            return [dict(row) for row in rows]

    def finish(self, item_id, status, bytes_=0, sha256=None, error=None):
        with self._lock:
            self._db.execute(
                "UPDATE items SET status = ?, bytes = ?, sha256 = ?, error = ?,"
                " attempts = attempts + 1, updated_at = ? WHERE id = ?",
                (status, bytes_, sha256, error, time.time(), item_id),
            )
            self._db.commit()

    def release(self, item_id, error=None):
        """Hand an active row back to pending -- used when a worker is interrupted mid-flight."""
        with self._lock:
            self._db.execute(
                "UPDATE items SET status = 'pending', error = ?, updated_at = ? WHERE id = ?",
                (error, time.time(), item_id),
            )
            self._db.commit()

    def requeue_active(self):
        """Startup repair: nothing can legitimately be active before the first worker starts."""
        with self._lock:
            cursor = self._db.execute(
                "UPDATE items SET status = 'pending' WHERE status = 'active'")
            self._db.commit()
            return cursor.rowcount

    # A dead URL and a slow one fail differently, and only one of them is worth asking again.
    # 403/404/410 and "undecodable" are verdicts about the resource; timeouts, dropped connections
    # and 5xx are verdicts about a moment.
    TRANSIENT = ("timeout", "ConnectionError", "SSLError", "ChunkedEncodingError",
                 "ReadTimeout", "ConnectTimeout", "interrupted",
                 "http 429", "http 500", "http 502", "http 503", "http 504",
                 "http 520", "http 521", "http 522", "http 523", "http 524", "http 530")

    def retry_transient(self, source):
        """Requeue only the failures that might succeed on a second look. Returns how many.

        This is what keeps LAION's validation split viable: it ships just 1.9x spare capacity
        against its target, and at the measured 51% first-pass success rate that lands ~700 images
        short. Most of the shortfall is timeouts, not dead links.
        """
        clauses = " OR ".join(["error LIKE ?"] * len(self.TRANSIENT))
        with self._lock:
            cursor = self._db.execute(
                f"UPDATE items SET status = 'pending', error = NULL"
                f" WHERE source = ? AND status = 'failed' AND ({clauses})",
                (source, *[f"%{pattern}%" for pattern in self.TRANSIENT]),
            )
            self._db.commit()
            return cursor.rowcount

    def retry_failed(self, source=None):
        with self._lock:
            if source:
                cursor = self._db.execute(
                    "UPDATE items SET status = 'pending', error = NULL"
                    " WHERE status = 'failed' AND source = ?", (source,))
            else:
                cursor = self._db.execute(
                    "UPDATE items SET status = 'pending', error = NULL WHERE status = 'failed'")
            self._db.commit()
            return cursor.rowcount

    def counts(self):
        """{source: {status: n}} -- the dashboard's main number source."""
        with self._lock:
            rows = self._db.execute(
                "SELECT source, status, COUNT(*) AS n, COALESCE(SUM(bytes), 0) AS b"
                " FROM items GROUP BY source, status").fetchall()
        out = {}
        for row in rows:
            entry = out.setdefault(row["source"], {})
            entry[row["status"]] = row["n"]
            if row["status"] == DONE:
                entry["bytes"] = row["b"]
        return out

    def failure_reasons(self, source=None, limit=12):
        """Tallied failure reasons, in the same shape harvest() prints in imports/sample."""
        query = ("SELECT error, COUNT(*) AS n FROM items WHERE status = 'failed' AND error IS NOT NULL")
        params = []
        if source:
            query += " AND source = ?"
            params.append(source)
        query += " GROUP BY error ORDER BY n DESC LIMIT ?"
        params.append(limit)
        with self._lock:
            rows = self._db.execute(query, params).fetchall()
        return [(row["error"], row["n"]) for row in rows]

    def pending_exists(self, source):
        with self._lock:
            row = self._db.execute(
                "SELECT 1 FROM items WHERE source = ? AND status IN ('pending', 'active') LIMIT 1",
                (source,)).fetchone()
        return row is not None

    def done_file_ids(self, source):
        with self._lock:
            rows = self._db.execute(
                "SELECT file_id FROM items WHERE source = ? AND status = 'done'", (source,)).fetchall()
        return {row["file_id"] for row in rows}

    def done_by_split(self, source):
        """{split: completed count} -- how LAION knows when it has hit its target."""
        with self._lock:
            rows = self._db.execute(
                "SELECT split, COUNT(*) AS n FROM items WHERE source = ? AND status = 'done'"
                " GROUP BY split", (source,)).fetchall()
        return {row["split"]: row["n"] for row in rows}

    def items_for_manifest(self, source=None, split=None):
        query = "SELECT * FROM items WHERE status = 'done'"
        params = []
        if source:
            query += " AND source = ?"
            params.append(source)
        if split:
            query += " AND split = ?"
            params.append(split)
        with self._lock:
            return [dict(r) for r in self._db.execute(query, params).fetchall()]

    # -- artifacts -------------------------------------------------------------------------------

    def add_artifact(self, name, source, url, dest, expected_bytes=None, md5=None):
        with self._lock:
            self._db.execute(
                "INSERT OR IGNORE INTO artifacts"
                " (name, source, url, dest, expected_bytes, md5, status, updated_at)"
                " VALUES (?, ?, ?, ?, ?, ?, 'pending', ?)",
                (name, source, url, str(dest), expected_bytes, md5, time.time()),
            )
            self._db.commit()

    def artifact(self, name):
        with self._lock:
            row = self._db.execute("SELECT * FROM artifacts WHERE name = ?", (name,)).fetchone()
        return dict(row) if row else None

    def artifacts(self, source=None):
        query, params = "SELECT * FROM artifacts", []
        if source:
            query += " WHERE source = ?"
            params.append(source)
        with self._lock:
            return [dict(r) for r in self._db.execute(query, params).fetchall()]

    def update_artifact(self, name, **fields):
        if not fields:
            return
        fields["updated_at"] = time.time()
        assigns = ", ".join(f"{k} = ?" for k in fields)
        with self._lock:
            self._db.execute(f"UPDATE artifacts SET {assigns} WHERE name = ?",
                             [*fields.values(), name])
            self._db.commit()

    # -- telemetry -------------------------------------------------------------------------------

    def log(self, level, msg, source=None):
        with self._lock:
            self._db.execute("INSERT INTO events (ts, source, level, msg) VALUES (?, ?, ?, ?)",
                             (time.time(), source, level, msg))
            self._db.commit()
        print(f"[{level}] {source or '-':<8} {msg}", flush=True)

    def recent_events(self, limit=40):
        with self._lock:
            rows = self._db.execute(
                "SELECT * FROM events ORDER BY ts DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def record_bytes(self, source, n):
        with self._lock:
            self._db.execute("INSERT INTO throughput (ts, source, bytes) VALUES (?, ?, ?)",
                             (time.time(), source, n))
            self._db.commit()

    def recent_rate(self, window=30.0):
        """{source: bytes/s} over the last `window` seconds, plus a 'total' key."""
        cutoff = time.time() - window
        with self._lock:
            rows = self._db.execute(
                "SELECT source, COALESCE(SUM(bytes), 0) AS b FROM throughput WHERE ts > ?"
                " GROUP BY source", (cutoff,)).fetchall()
        rates = {row["source"]: row["b"] / window for row in rows}
        rates["total"] = sum(rates.values())
        return rates

    def prune_throughput(self, keep_seconds=600):
        with self._lock:
            self._db.execute("DELETE FROM throughput WHERE ts < ?", (time.time() - keep_seconds,))
            self._db.commit()

    def snapshot_json(self):
        return json.dumps({"counts": self.counts(), "control": self.all_control()}, indent=2)

    def close(self):
        with self._lock:
            self._db.close()


def reconcile(state, verbose=True):
    """Repair item status from what is actually on disk.

    Ground truth is the filesystem: images are written to `.part` and renamed, so a file that exists
    is complete. Anything marked done whose file has vanished goes back to pending, and anything on
    disk that the database does not know is done gets marked done. This is what makes a lost or
    corrupt database a rescan rather than a 30-hour re-download.
    """
    fixed_missing = fixed_found = 0
    with state._lock:
        rows = state._db.execute("SELECT id, dest, status FROM items").fetchall()

    to_pending, to_done = [], []
    for row in rows:
        path = config.DATA_ROOT / row["dest"]
        # A zero-byte file is not a download, it is the wreckage of one. Treat it as absent so it
        # gets refetched rather than silently counted as complete.
        present = path.exists() and path.stat().st_size > 0
        if row["status"] == DONE and not present:
            to_pending.append(row["id"])
        elif row["status"] in (PENDING, ACTIVE, FAILED) and present:
            to_done.append(row["id"])

    with state._lock:
        if to_pending:
            state._db.executemany(
                "UPDATE items SET status = 'pending', updated_at = ? WHERE id = ?",
                [(time.time(), i) for i in to_pending])
            fixed_missing = len(to_pending)
        if to_done:
            state._db.executemany(
                "UPDATE items SET status = 'done', updated_at = ? WHERE id = ?",
                [(time.time(), i) for i in to_done])
            fixed_found = len(to_done)
        state._db.commit()

    if verbose:
        print(f"reconcile: {fixed_found} found on disk marked done, "
              f"{fixed_missing} missing files requeued")
    return fixed_found, fixed_missing
