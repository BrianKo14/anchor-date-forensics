#!/usr/bin/env bash
# Start (or attach to) the unattended scoring run on the lab server.
#
#   ./runner/run.sh start <manifest> <out-dir>   launch detached; survives logout
#   ./runner/run.sh smoke <manifest> <out-dir>   ~5 min rehearsal of the whole path
#   ./runner/run.sh attach                       watch the console output
#   ./runner/run.sh status                       is it alive, and how far along
#   ./runner/run.sh stop                         ask it to stop cleanly (resumable)
#   ./runner/run.sh merge <manifest> <out-dir>   assemble finished chunks, then exit
#
# The job runs inside tmux with setsid+nohup, so init owns it rather than your SSH session:
# closing the laptop, dropping the VPN or `exit`ing the shell does not touch it. Only *viewing*
# the dashboard needs a connection, because it binds to 127.0.0.1 deliberately -- tunnel it with
#
#     ssh -F ssh_config -L 8766:127.0.0.1:8766 remote
#
# and open http://localhost:8766. Close the tunnel whenever; the run keeps going.

set -euo pipefail

SESSION="${PANEL_SESSION:-panel-scoring}"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PANEL_PYTHON:-$HERE/.venv/bin/python}"
LOGDIR="${PANEL_VAR_DIR:-$HERE/var}/logs"

mkdir -p "$LOGDIR"

running() { tmux has-session -t "$SESSION" 2>/dev/null; }

usage() { sed -n '2,20p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
  start|smoke)
    [[ $# -ge 3 ]] || usage
    manifest="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    outdir="$3"
    [[ -f "$manifest" ]] || { echo "no manifest at $manifest" >&2; exit 1; }
    if running; then
      echo "already running in tmux session '$SESSION' -- use 'attach' to watch it"
      exit 0
    fi
    if [[ ! -x "$PYTHON" ]]; then
      echo "no interpreter at $PYTHON" >&2
      exit 1
    fi
    args="--manifest '$manifest' --out-dir '$outdir'"
    [[ "$1" == "smoke" ]] && args="$args --smoke --limit ${4:-40}"
    LOG="$LOGDIR/scoring-$(date +%Y%m%d-%H%M%S).log"

    # setsid detaches tmux from this SSH session's process group, so even a hard disconnect
    # cannot take the run down with it.
    setsid nohup tmux new-session -d -s "$SESSION" \
      "cd '$HERE' && '$PYTHON' -m runner.pipeline $args 2>&1 | tee '$LOG'" \
      >/dev/null 2>&1 </dev/null

    sleep 3
    if running; then
      echo "started in tmux session '$SESSION'"
      echo "  log        $LOG"
      echo "  dashboard  ssh -F ssh_config -L 8766:127.0.0.1:8766 remote  ->  http://localhost:8766"
      echo "  watch      ./runner/run.sh attach     (Ctrl-B D to leave it running)"
    else
      echo "failed to start; last lines of $LOG:" >&2
      tail -20 "$LOG" >&2 || true
      exit 1
    fi
    ;;

  attach)
    running || { echo "not running"; exit 1; }
    echo "Ctrl-B then D detaches without stopping the run."
    exec tmux attach -t "$SESSION"
    ;;

  status)
    if running; then echo "running in tmux session '$SESSION'"; else echo "not running"; fi
    latest="$(ls -t "$LOGDIR"/scoring-*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "--- tail of $latest ---"; tail -20 "$latest"; }
    ;;

  stop)
    running || { echo "not running"; exit 0; }
    # Signal the interpreter, not the shell wrapping it: `pgrep -f runner.pipeline` also matches
    # the tmux command string, and killing that leaves the real run orphaned and still going.
    pid="$(pgrep -f '[p]ython -m runner\.pipeline' | head -1)"
    if [[ -z "$pid" ]]; then
      echo "tmux session is up but no pipeline process found; killing the session"
      tmux kill-session -t "$SESSION"
      exit 0
    fi
    # SIGTERM, not SIGKILL: the pipeline catches it, lets the chunks in flight finish, and leaves
    # everything else pending so the next start resumes instead of redoing work.
    kill -TERM "$pid"
    echo -n "stopping (finishing chunks in flight)"
    for _ in $(seq 1 180); do
      kill -0 "$pid" 2>/dev/null || break
      echo -n "."; sleep 1
    done
    echo
    if kill -0 "$pid" 2>/dev/null; then
      echo "still shutting down after 3 min -- leaving it; check './runner/run.sh status'"
    else
      echo "stopped cleanly. finished chunks kept; restart with './runner/run.sh start ...'"
    fi
    ;;

  merge)
    [[ $# -ge 3 ]] || usage
    exec "$PYTHON" -m runner.pipeline --manifest "$2" --out-dir "$3" --merge-only
    ;;

  *) usage ;;
esac
