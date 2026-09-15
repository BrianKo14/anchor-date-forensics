"""HTTP fetching: small images in memory, large files with byte-range resume.

Two shapes of download, because the two failure modes are different:

`fetch_bytes` pulls one image into memory. Failures here are *expected* -- roughly a fifth of the
LAION URLs are dead -- so it returns a reason instead of raising, exactly as
`imports/sample/authentic.py::fetch` does, and the caller tallies the reasons.

`download_artifact` pulls a multi-hundred-megabyte file to disk and resumes from wherever a previous
attempt stopped. Both COCO's S3 bucket and RAISE's host were verified to answer `Range` with HTTP
206, which is what makes a killed 18 GB download cost seconds rather than hours.

Everything is written to `<dest>.part` and renamed with `os.replace` only once complete, so a file
that exists on disk is always whole. That invariant is what `state.reconcile` relies on.
"""

import hashlib
import os
import threading
import time

import requests

import config

# One session per thread. requests.Session is not documented as thread-safe, and its connection
# pool is per-session, so sharing one across sixteen LAION workers means them queueing on each
# other's sockets. Created at import time, not lazily: a `global` guard would itself be a race.
_local = threading.local()


def session():
    if not hasattr(_local, "session"):
        _local.session = requests.Session()
        _local.session.headers["User-Agent"] = config.USER_AGENT
    return _local.session


def fetch_bytes(url, bucket=None, stop=None, read_timeout=None, max_bytes=None, on_progress=None):
    """GET one resource into memory. Returns (content, reason); content is None on failure.

    Streams rather than using `response.content` so the token bucket sees the bytes as they arrive
    and a rate limit actually limits the rate, instead of throttling only between whole images.
    """
    timeout = (config.CONNECT_TIMEOUT, read_timeout or config.READ_TIMEOUT)
    try:
        response = session().get(url, timeout=timeout, stream=True)
        response.raise_for_status()
        chunks, total = [], 0
        for chunk in response.iter_content(config.CHUNK_BYTES):
            if stop is not None and stop.is_set():
                return None, "interrupted"
            if not chunk:
                continue
            if bucket is not None and not bucket.consume(len(chunk), stop=stop):
                return None, "interrupted"
            chunks.append(chunk)
            total += len(chunk)
            if on_progress is not None:
                on_progress(len(chunk))
            if max_bytes and total > max_bytes:
                return None, f"over {max_bytes} bytes"
        return b"".join(chunks), "ok"
    except requests.HTTPError as error:
        return None, f"http {error.response.status_code}"
    except requests.Timeout:
        return None, "timeout"
    except Exception as error:  # noqa: BLE001 -- the reason string is the product here
        return None, type(error).__name__


def head(url):
    """(status, content_length, etag). Used to size and verify artifacts before pulling them."""
    try:
        response = session().head(url, timeout=(config.CONNECT_TIMEOUT, config.READ_TIMEOUT),
                                  allow_redirects=True)
        length = response.headers.get("x-linked-size") or response.headers.get("content-length")
        etag = (response.headers.get("etag") or "").strip('"')
        return response.status_code, int(length) if length else None, etag
    except Exception:  # noqa: BLE001
        return None, None, None


def download_artifact(url, dest, expected_bytes=None, bucket=None, stop=None,
                      on_progress=None, attempts=None, connections=1, gate=None):
    """Resumable download to `dest`. Returns (ok, bytes_written, reason).

    With `connections > 1` the file is split into that many contiguous byte ranges fetched in
    parallel and concatenated. This is not premature optimisation: measured against COCO's S3
    bucket, a single stream took 204 KB/s while a second concurrent request independently pulled
    278 KB/s more, so one connection was leaving over half the available throughput unused on the
    largest transfer in the job (18.8 GiB).

    Falls back to a single stream when the size is unknown or the server will not honour ranges.
    """
    if connections > 1 and expected_bytes:
        return _download_segmented(url, dest, expected_bytes, bucket, stop, on_progress,
                                   attempts or config.MAX_ATTEMPTS, connections, gate)
    return _download_sequential(url, dest, expected_bytes, bucket, stop, on_progress, attempts, gate)


def _segment_bounds(total, count):
    """Contiguous [start, end] byte ranges covering the whole file, last one absorbing the remainder."""
    step = total // count
    bounds = []
    for index in range(count):
        start = index * step
        end = (total - 1) if index == count - 1 else (start + step - 1)
        bounds.append((start, end))
    return bounds


def _fetch_segment(url, part_path, start, end, bucket, stop, on_progress, attempts, gate=None):
    """One segment, itself resumable from whatever is already in its part file."""
    want = end - start + 1
    for attempt in range(attempts):
        have = os.path.getsize(part_path) if os.path.exists(part_path) else 0
        if have >= want:
            return True, "ok"
        if stop is not None and stop.is_set():
            return False, "interrupted"
        if attempt and stop is not None and stop.wait(config.BACKOFF_BASE ** attempt):
            return False, "interrupted"

        try:
            response = session().get(
                url, headers={"Range": f"bytes={start + have}-{end}"}, stream=True,
                timeout=(config.CONNECT_TIMEOUT, config.READ_TIMEOUT))
            if response.status_code != 206:
                # No range support: the caller must fall back, and appending a whole body onto a
                # segment would corrupt it.
                response.close()
                return False, f"no range support (http {response.status_code})"

            written = have  # tracked explicitly; file.tell() on an append handle is not portable
            with open(part_path, "ab") as handle:
                for chunk in response.iter_content(config.CHUNK_BYTES):
                    if stop is not None and stop.is_set():
                        return False, "interrupted"
                    if not chunk:
                        continue
                    if gate is not None and not gate():
                        return False, "interrupted"
                    if bucket is not None and not bucket.consume(len(chunk), stop=stop):
                        return False, "interrupted"
                    # Never overshoot: a server that ignored the range *end* would otherwise write
                    # past this segment and into the next one's territory, and the corruption would
                    # only surface as a bad archive hours later.
                    room = want - written
                    if room <= 0:
                        break
                    chunk = chunk[:room]
                    handle.write(chunk)
                    written += len(chunk)
                    if on_progress is not None:
                        on_progress(len(chunk))

            if os.path.getsize(part_path) >= want:
                return True, "ok"
        except Exception as error:  # noqa: BLE001
            last = type(error).__name__
            if attempt == attempts - 1:
                return False, last
    return False, "incomplete after retries"


def _download_segmented(url, dest, expected_bytes, bucket, stop, on_progress, attempts,
                        connections, gate=None):
    dest = os.fspath(dest)
    if os.path.exists(dest) and os.path.getsize(dest) == expected_bytes:
        return True, expected_bytes, "already on disk"

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    bounds = _segment_bounds(expected_bytes, connections)
    parts = [f"{dest}.part{index}" for index in range(connections)]

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(max_workers=connections) as pool:
        futures = [
            pool.submit(_fetch_segment, url, parts[i], start, end, bucket, stop, on_progress,
                        attempts, gate)
            for i, (start, end) in enumerate(bounds)
        ]
        results = [future.result() for future in futures]

    if not all(ok for ok, _ in results):
        reasons = {reason for ok, reason in results if not ok}
        if any("no range support" in r for r in reasons):
            for path in parts:
                if os.path.exists(path):
                    os.remove(path)
            return _download_sequential(url, dest, expected_bytes, bucket, stop, on_progress,
                                        attempts, gate)
        return False, sum(os.path.getsize(p) for p in parts if os.path.exists(p)), "; ".join(reasons)

    # Concatenate in order, then rename. Assembling into a temporary file means an interrupted
    # concatenation cannot leave a half-built archive sitting at the destination path.
    assembled = dest + ".assembling"
    with open(assembled, "wb") as out:
        for path in parts:
            with open(path, "rb") as segment:
                while True:
                    block = segment.read(1 << 22)
                    if not block:
                        break
                    out.write(block)

    size = os.path.getsize(assembled)
    if size != expected_bytes:
        os.remove(assembled)
        return False, size, f"assembled {size} != expected {expected_bytes}"

    os.replace(assembled, dest)
    for path in parts:
        os.remove(path)
    return True, size, "ok"


def _download_sequential(url, dest, expected_bytes=None, bucket=None, stop=None,
                         on_progress=None, attempts=None, gate=None):
    """Single-stream resumable download.

    Resumes by asking for `Range: bytes=<size of the .part file>-`. If the server ignores the range
    and answers 200 rather than 206, the partial file is discarded and the download restarts -- the
    alternative, appending a full body onto a partial one, would silently corrupt the file.
    """
    dest = os.fspath(dest)
    part = dest + ".part"
    attempts = attempts or config.MAX_ATTEMPTS

    if os.path.exists(dest):
        size = os.path.getsize(dest)
        if expected_bytes is None or size == expected_bytes:
            return True, size, "already on disk"
        os.remove(dest)  # wrong size: a truncated or stale file is worse than no file

    os.makedirs(os.path.dirname(dest) or ".", exist_ok=True)
    last_reason = "unknown"

    for attempt in range(attempts):
        if stop is not None and stop.is_set():
            return False, 0, "interrupted"
        if attempt:
            delay = config.BACKOFF_BASE ** attempt
            if stop is not None and stop.wait(delay):
                return False, 0, "interrupted"
            elif stop is None:
                time.sleep(delay)

        have = os.path.getsize(part) if os.path.exists(part) else 0
        headers = {"Range": f"bytes={have}-"} if have else {}

        try:
            response = session().get(url, headers=headers, stream=True,
                                     timeout=(config.CONNECT_TIMEOUT, config.READ_TIMEOUT))

            if have and response.status_code == 200:
                # Range ignored: start over rather than append onto the partial file.
                response.close()
                os.remove(part)
                last_reason = "range ignored, restarting"
                continue
            if have and response.status_code == 416:
                # Already have everything the server is willing to give.
                response.close()
                os.replace(part, dest)
                return True, os.path.getsize(dest), "ok"
            response.raise_for_status()

            mode = "ab" if (have and response.status_code == 206) else "wb"
            if mode == "wb":
                have = 0

            with open(part, mode) as handle:
                for chunk in response.iter_content(config.CHUNK_BYTES):
                    if stop is not None and stop.is_set():
                        handle.flush()
                        return False, have, "interrupted"
                    if not chunk:
                        continue
                    if gate is not None and not gate():
                        handle.flush()
                        return False, have, "interrupted"
                    if bucket is not None and not bucket.consume(len(chunk), stop=stop):
                        handle.flush()
                        return False, have, "interrupted"
                    handle.write(chunk)
                    have += len(chunk)
                    if on_progress is not None:
                        on_progress(len(chunk))

            if expected_bytes is not None and have != expected_bytes:
                last_reason = f"size {have} != expected {expected_bytes}"
                continue

            os.replace(part, dest)
            return True, have, "ok"

        except requests.HTTPError as error:
            last_reason = f"http {error.response.status_code}"
        except requests.Timeout:
            last_reason = "timeout"
        except Exception as error:  # noqa: BLE001
            last_reason = type(error).__name__

    return False, 0, last_reason


def sha256_bytes(data):
    return hashlib.sha256(data).hexdigest()


def sha256_file(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def md5_file(path, chunk=1 << 20):
    digest = hashlib.md5()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def write_atomic(path, data):
    """Write bytes so the file never exists in a partial state. Returns its sha256."""
    path = os.fspath(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    part = path + ".part"
    with open(part, "wb") as handle:
        handle.write(data)
    os.replace(part, path)
    return sha256_bytes(data)
