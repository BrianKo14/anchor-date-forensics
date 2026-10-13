#!/usr/bin/env bash
# Start (or attach to) the unattended scoring run on the lab server.
#
#   ./runner/run.sh start <manifest> <out-dir> [args...]   launch detached; survives logout
#   ./runner/run.sh smoke <manifest> <out-dir> [limit]     rehearsal of the whole path
#
# Anything after <out-dir> on `start` is passed through to runner.pipeline, e.g.
#   ./runner/run.sh start m.csv out/ --detectors cnndetection univfd
#   ./runner/run.sh attach                       watch the console output
#   ./runner/run.sh status                       is it alive, and how far along
#   ./runner/run.sh stop                         ask it to stop cleanly (resumable)
#   ./runner/run.sh merge <manifest> <out-dir>   assemble finished chunks, then exit
#
# Prefix any of these with `selfdesc` to drive the self-description job instead (CPU, main's
# selfdesc/: extraction of every config, then the temporal tests; runner/selfdesc.py):
#   ./runner/run.sh selfdesc smoke <manifest> <out-dir>   rehearse every stage on ~470 images
#   ./runner/run.sh selfdesc start <manifest> <out-dir>   the real run, detached
#   ./runner/run.sh selfdesc queue <queue-file>      several runs back to back (selfdesc_queue.py)
#   ./runner/run.sh selfdesc queue-smoke <queue-file>  every run of the queue as a smoke run
#   ./runner/run.sh selfdesc status | attach | stop
# It has its own tmux session (selfdesc), job database and dashboard port (8767), so it can run
# alongside a panel run.
#
# The job runs inside tmux with setsid+nohup, so init owns it rather than your SSH session:
# closing the laptop, dropping the VPN or `exit`ing the shell does not touch it. Only *viewing*
# the dashboard needs a connection, because it binds to 127.0.0.1 deliberately -- tunnel it with
#
#     ssh -F ssh_config -L 8766:127.0.0.1:8766 remote
#
# and open http://localhost:8766. Close the tunnel whenever; the run keeps going.

set -euo pipefail

JOB=panel
if [[ "${1:-}" == "selfdesc" ]]; then JOB=selfdesc; shift; fi
if [[ "$JOB" == panel ]]; then
  SESSION="${PANEL_SESSION:-panel-scoring}"; MODULE=runner.pipeline; PREFIX=scoring
  PORT="${PANEL_PORT:-8766}"
else
  SESSION="${SELFDESC_SESSION:-selfdesc}"; MODULE=runner.selfdesc; PREFIX=selfdesc
  PORT="${SELFDESC_PORT:-8767}"
fi
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${PANEL_PYTHON:-$HERE/.venv/bin/python}"
LOGDIR="${PANEL_VAR_DIR:-$HERE/var}/logs"

mkdir -p "$LOGDIR"

running() { tmux has-session -t "$SESSION" 2>/dev/null; }

abspath() { case "$1" in /*) echo "$1" ;; *) echo "$PWD/$1" ;; esac; }

usage() { sed -n '2,30p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 1; }

case "${1:-}" in
  start|smoke)
    [[ $# -ge 3 ]] || usage
    manifest="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    # Absolute before the `cd $HERE` below: a relative out-dir means the caller's directory, and
    # the outputs belong in main's experiments/, not in this branch's checkout.
    outdir="$(abspath "$3")"
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
    if [[ "$1" == "smoke" && "$JOB" == selfdesc ]]; then
      args="$args --smoke"
    elif [[ "$1" == "smoke" ]]; then
      args="$args --smoke --limit ${4:-40}"
    else
      shift 3
      for extra in "$@"; do args="$args '$extra'"; done
    fi
    LOG="$LOGDIR/$PREFIX-$(date +%Y%m%d-%H%M%S).log"

    # setsid detaches tmux from this SSH session's process group, so even a hard disconnect
    # cannot take the run down with it.
    setsid nohup tmux new-session -d -s "$SESSION" \
      "cd '$HERE' && '$PYTHON' -m $MODULE $args 2>&1 | tee '$LOG'" \
      >/dev/null 2>&1 </dev/null

    sleep 3
    if running; then
      echo "started in tmux session '$SESSION'"
      echo "  log        $LOG"
      echo "  dashboard  ssh -F ssh_config -L $PORT:127.0.0.1:$PORT remote  ->  http://localhost:$PORT"
      echo "  watch      ./runner/run.sh ${JOB/panel/} attach     (Ctrl-B D to leave it running)"
    else
      echo "failed to start; last lines of $LOG:" >&2
      tail -20 "$LOG" >&2 || true
      exit 1
    fi
    ;;

  queue|queue-smoke)
    [[ "$JOB" == selfdesc && $# -ge 2 ]] || usage
    queue="$(cd "$(dirname "$2")" && pwd)/$(basename "$2")"
    [[ -f "$queue" ]] || { echo "no queue file at $queue" >&2; exit 1; }
    if running; then
      echo "already running in tmux session '$SESSION' -- use 'attach' to watch it"
      exit 0
    fi
    args="'$queue'"
    [[ "$1" == "queue-smoke" ]] && args="$args --smoke"
    LOG="$LOGDIR/$PREFIX-$(date +%Y%m%d-%H%M%S).log"
    setsid nohup tmux new-session -d -s "$SESSION" \
      "cd '$HERE' && '$PYTHON' -m runner.selfdesc_queue $args 2>&1 | tee '$LOG'" \
      >/dev/null 2>&1 </dev/null
    sleep 3
    if running; then
      echo "queue started in tmux session '$SESSION'"
      echo "  log        $LOG"
      echo "  dashboard  ssh -F ssh_config -L $PORT:127.0.0.1:$PORT remote  ->  http://localhost:$PORT"
      echo "  watch      ./runner/run.sh selfdesc attach     (Ctrl-B D to leave it running)"
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
    latest="$(ls -t "$LOGDIR"/$PREFIX-*.log 2>/dev/null | head -1 || true)"
    [[ -n "$latest" ]] && { echo "--- tail of $latest ---"; tail -20 "$latest"; }
    ;;

  stop)
    running || { echo "not running"; exit 0; }
    # Signal the interpreter, not the shell wrapping it: `pgrep -f runner.pipeline` also matches
    # the tmux command string, and killing that leaves the real run orphaned and still going.
    pid="$(pgrep -f "[p]ython -m ${MODULE//./\\.}" | head -1)"
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
    manifest="$(abspath "$2")"; outdir="$(abspath "$3")"
    cd "$HERE"
    if [[ "$JOB" == selfdesc ]]; then
      exec "$PYTHON" -m runner.selfdesc --manifest "$manifest" --out-dir "$outdir" --assemble-only
    fi
    exec "$PYTHON" -m runner.pipeline --manifest "$manifest" --out-dir "$outdir" --merge-only
    ;;

  *) usage ;;
esac
