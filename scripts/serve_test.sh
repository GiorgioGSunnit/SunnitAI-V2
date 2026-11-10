#!/usr/bin/env bash
# The interactive flow for a hand test (the PO drives h3): uvicorn from this checkout (/root/v38) on a spare port, with
# the server's .env.v45 plus INTERACTIVE_FLOW, TOPIC_STEP and TOPIC_TABLES on. Nothing in /opt/chatbot is read or
# written, the chatbot service is not touched, no LLM setting is overridden (the model is .env.v45's).
#
#   scripts/serve_test.sh [start] [PORT]   start (default: the first free port of 8030-8049; never 8010)
#   scripts/serve_test.sh stop       stop the server this script started
#   scripts/serve_test.sh status     is it up, on which port
#
# HOST=0.0.0.0 to reach it from outside (default 127.0.0.1: ssh -L PORT:127.0.0.1:PORT root@<server>);
# ENV_FILE=... another env file; PY=... another python (default venv/bin/python, else /opt/chatbot/venv/bin/python:
# only its interpreter is used, the code is this checkout's).
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$(pwd)"
case "$ROOT" in
    /opt/chatbot*) echo "refusing: this is $ROOT, run it from /root/v38"; exit 1 ;;
esac
ENV_FILE="${ENV_FILE:-.env.v45}"
if [ -z "${PY:-}" ]; then
    if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=/opt/chatbot/venv/bin/python; fi
fi
HOST="${HOST:-127.0.0.1}"
PIDFILE="$ROOT/serve_test.pid"
LOG="$ROOT/serve_test.log"

running() { [ -f "$PIDFILE" ] && kill -0 "$(cut -d' ' -f1 "$PIDFILE")" 2>/dev/null; }
in_use() { (ss -ltn 2>/dev/null || netstat -ltn 2>/dev/null) | awk '{print $4}' | grep -Eq "[:.]$1\$"; }

case "${1:-}" in
    stop)
        if running; then kill "$(cut -d' ' -f1 "$PIDFILE")" && echo "stopped (port $(cut -d' ' -f2 "$PIDFILE"))"; fi
        rm -f "$PIDFILE"
        exit 0 ;;
    status)
        if running; then echo "up on port $(cut -d' ' -f2 "$PIDFILE") (pid $(cut -d' ' -f1 "$PIDFILE")), log $LOG"
        else echo "not running"; fi
        exit 0 ;;
    start) shift ;;
esac

if running; then
    echo "already up on port $(cut -d' ' -f2 "$PIDFILE"): scripts/serve_test.sh stop first"
    exit 1
fi
[ -f "$ENV_FILE" ] || { echo "no $ENV_FILE in $ROOT"; exit 1; }
[ -x "$PY" ] || { echo "no python at $PY"; exit 1; }

PORT="${1:-}"
case "$PORT" in
    ''|*[!0-9]*) [ -z "$PORT" ] || { echo "usage: scripts/serve_test.sh [start] [PORT] | stop | status"; exit 1; } ;;
esac
if [ -z "$PORT" ]; then
    for p in $(seq 8030 8049); do
        if ! in_use "$p"; then PORT="$p"; break; fi
    done
fi
[ -n "$PORT" ] || { echo "no free port in 8030-8049"; exit 1; }
[ "$PORT" != "8010" ] || { echo "refusing port 8010"; exit 1; }
if in_use "$PORT"; then echo "port $PORT is in use"; exit 1; fi

# the env file's KEY=VALUE lines (comments and blanks skipped, surrounding quotes removed), then the three flags
while IFS= read -r line || [ -n "$line" ]; do
    line="${line%$'\r'}"
    [[ "$line" =~ ^[[:space:]]*(#|$) ]] && continue
    [[ "$line" =~ ^[[:space:]]*(export[[:space:]]+)?([A-Za-z_][A-Za-z0-9_]*)=(.*)$ ]] || continue
    key="${BASH_REMATCH[2]}"; value="${BASH_REMATCH[3]}"
    value="${value%\"}"; value="${value#\"}"; value="${value%\'}"; value="${value#\'}"
    export "$key=$value"
done < "$ENV_FILE"
export INTERACTIVE_FLOW=true TOPIC_STEP=true TOPIC_TABLES=true

echo "== serve_test $(date), code $(git rev-parse --short HEAD 2>/dev/null || echo '?'), $ENV_FILE + INTERACTIVE_FLOW," \
     "TOPIC_STEP, TOPIC_TABLES, port $PORT" > "$LOG"
nohup "$PY" -m uvicorn src.chatbot.api:app --host "$HOST" --port "$PORT" >> "$LOG" 2>&1 &
echo "$! $PORT" > "$PIDFILE"

for _ in $(seq 1 90); do
    if curl -sf "http://127.0.0.1:$PORT/api/health" > /dev/null 2>&1; then
        echo "up: http://$HOST:$PORT (POST /api/chat; log $LOG; stop: scripts/serve_test.sh stop)"
        [ "$HOST" = "127.0.0.1" ] && echo "from a laptop: ssh -L $PORT:127.0.0.1:$PORT root@<server>, then http://localhost:$PORT"
        exit 0
    fi
    running || { echo "the server exited: tail $LOG"; tail -20 "$LOG"; rm -f "$PIDFILE"; exit 1; }
    sleep 2
done
echo "no answer on /api/health after 180 s: tail $LOG (still running: scripts/serve_test.sh stop to end it)"
exit 1
