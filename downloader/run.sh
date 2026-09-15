#!/usr/bin/env bash
# Start (or attach to) the unattended download on the lab server.
#
#   ./downloader/run.sh start     launch detached; returns immediately, survives logout
#   ./downloader/run.sh attach    watch the console output
#   ./downloader/run.sh status    is it alive, and how far along
#   ./downloader/run.sh stop      ask it to stop cleanly (resumable; nothing is lost)
#   ./downloader/run.sh smoke     ~3 min rehearsal of the whole path
#
# The job runs inside tmux with setsid+nohup, so it is owned by init rather than by your SSH
# session: closing the laptop, dropping the VPN, or `exit`ing the shell does not touch it. The only
# thing that needs an SSH connection is *viewing* the dashboard, because it deliberately binds to
# 127.0.0.1 -- tunnel it with
#
#     ssh -F ssh_config -L 8765:127.0.0.1:8765 remote
#
# and open http://localhost:8765. Close the tunnel whenever; the download keeps going.

set -euo pipefail

SESSION="${AIGENBENCH_SESSION:-aigenbench}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${AIGENBENCH_PYTHON:-$HERE/.venv/bin/python}"
LOGDIR="${AIGENBENCH_VAR_DIR:-$HERE/var}/logs"
LOG="$LOGDIR/download-$(date +%Y%m%d-%H%M%S).log"

mkdir -p "$LOGDIR"

if [[ ! -x "$PYTHON" ]]; then
  echo "no interpreter at $PYTHON" >&2
  echo "create it with:  /usr/local/bin/python3.12 -m venv $HERE/.venv" >&2
  echo "                 $HERE/.venv/bin/pip install -r $HERE/requirements.txt" >&2
  exit 1
fi

running() { tmux has-session -t "$SESSION" 2>/dev/null; }

case "${1:-start}" in
  start|smoke)
    if running; then
      echo "already running in tmux session '$SESSION' -- use 'attach' to watch it"
      exit 0
    fi
    args=""
    [[ "$1" == "smoke" ]] && args="--smoke --limit ${2:-20}"

    # setsid detaches tmux from this SSH session's process group, so even a hard disconnect (as
    # opposed to a clean logout) cannot take the server down with it.
    setsid nohup tmux new-session -d -s "$SESSION" \
      "cd '$HERE' && '$PYTHON' -m downloader.pipeline $args 2>&1 | tee '$LOG'" \
      >/dev/null 2>&1 < /dev/null

    sleep 2
    if running; then
      echo "started in tmux session '$SESSION'"
      echo "  log        $LOG"
      echo "  dashboard  ssh -F ssh_config -L 8765:127.0.0.1:8765 remote  ->  http://localhost:8765"
      echo "  watch      ./downloader/run.sh attach     (Ctrl-B D to leave it running)"
    else
      echo "failed to start; last lines of $LOG:" >&2
      tail -20 "$LOG" >&2 || true
      exit 1
    fi
    ;;

  attach)
    running || { echo "not running"; exit 1; }
    echo "Ctrl-B then D detaches without stopping the job."
    exec tmux attach -t "$SESSION"
    ;;

  status)
    if running; then
      echo "running in tmux session '$SESSION'"
    else
      echo "not running"
    fi
    latest="$(ls -t "$LOGDIR"/download-*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "--- tail of $latest ---"; tail -15 "$latest"; }
    ;;

  stop)
    running || { echo "not running"; exit 0; }
    # Signal the interpreter itself, not the shell wrapping it. `pgrep -f downloader.pipeline` also
    # matches the tmux command string and any shell that mentions it, and killing those leaves the
    # actual download orphaned and still running.
    pid="$(pgrep -f '[p]ython -m downloader\.pipeline' | head -1)"
    if [[ -z "$pid" ]]; then
      echo "tmux session is up but no pipeline process found; killing the session"
      tmux kill-session -t "$SESSION"
      exit 0
    fi
    # SIGTERM, not SIGKILL: the pipeline catches it, lets in-flight writes finish, and returns
    # active rows to pending so the next start resumes instead of redoing work.
    kill -TERM "$pid"
    echo -n "stopping"
    for _ in $(seq 1 60); do
      kill -0 "$pid" 2>/dev/null || break
      echo -n "."; sleep 1
    done
    echo
    if kill -0 "$pid" 2>/dev/null; then
      echo "still shutting down after 60s -- leaving it; check with './downloader/run.sh status'"
    else
      echo "stopped cleanly. partial downloads kept; restart with './downloader/run.sh start'"
    fi
    ;;

  *)
    sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'
    exit 1
    ;;
esac
