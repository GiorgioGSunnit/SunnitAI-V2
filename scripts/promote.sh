#!/usr/bin/env bash
# v77: the test copy -> the live copy, on the server: /root/v38 -> /opt/chatbot. In order:
#
#   1. snapshot   /opt/chatbot's src/ scripts/ docs/ tests/ data/ state/ VERSION .env -> /root/snapshots/<time>_<version>/
#   2. rsync      per directory: src/ scripts/ docs/ tests/ mirrored (--delete), data/ added and updated only (never
#                 --delete), VERSION; scripts/deploy_exclude.txt never copied nor deleted (env files, venv, state/,
#                 heldout/, logs, the runtime files data/ used to hold)
#   3. flags      the boolean flags of /root/v38/.env.v45 (KEY=true|false|yes|no|on|off, KEY not LLM_*, EMBEDDING_*,
#                 NEO4J_*, STRIPE_*) set in /opt/chatbot/.env, each change printed; nothing else of .env is touched
#   4. import     scripts/import_check.py with /opt/chatbot/.env -- on failure the snapshot is put back, no restart
#   5. restart    stop, then the runtime state moved to /opt/chatbot/state/ once (data/sessions.json,
#                 data/tenant_logos/, data/system_templates/, data/article_cards.jsonl, data/logs/ and logs/ -- the
#                 service stopped, so no write is lost), then start
#   6. report     VERSION, systemctl is-active, GET /api/version
#
#   bash /root/v38/scripts/promote.sh
#   bash /root/v38/scripts/promote.sh --rollback /root/snapshots/<dir>    the snapshot back, then the restart
# v79.19.1 (item 0): refused (exit 1, nothing changed) when /root/v38/data or /opt/chatbot/data is a symlink.
#
# SRC, DST, SERVICE (chatbot), CHATBOT_URL (http://127.0.0.1:8000), SNAPSHOTS (/root/snapshots) override.
set -euo pipefail

SRC="${SRC:-/root/v38}"
DST="${DST:-/opt/chatbot}"
SERVICE="${SERVICE:-chatbot}"
CHATBOT_URL="${CHATBOT_URL:-http://127.0.0.1:8000}"
SNAPSHOTS="${SNAPSHOTS:-/root/snapshots}"
DIRS_MIRRORED="src scripts docs tests"
DIRS_ADDED="data"
SNAP_ITEMS="src scripts docs tests data state VERSION .env"
FLAG_SOURCE="$SRC/.env.v45"
FLAG_RE='^([A-Z][A-Z0-9_]*)=["'"'"']?([Tt][Rr][Uu][Ee]|[Ff][Aa][Ll][Ss][Ee]|[Yy][Ee][Ss]|[Nn][Oo]|[Oo][Nn]|[Oo][Ff][Ff])["'"'"']?[[:space:]]*$'
FLAG_SKIP_RE='^(LLM_|EMBEDDING_|NEO4J_|STRIPE_)'
EXCLUDE="$SRC/scripts/deploy_exclude.txt"
PY="$DST/venv/bin/python"

restore() {         # the snapshot $1 back into $DST: the mirrored directories exactly, the others added over
    local snap="$1"
    [ -d "$snap" ] || { echo "no snapshot $snap"; exit 1; }
    echo "== restoring $snap -> $DST"
    for d in $DIRS_MIRRORED; do
        [ -d "$snap/$d" ] && rsync -a --delete --exclude-from="$EXCLUDE" "$snap/$d/" "$DST/$d/"
    done
    for item in data state; do
        [ -d "$snap/$item" ] && rsync -a "$snap/$item/" "$DST/$item/"
    done
    for f in VERSION .env; do
        [ -f "$snap/$f" ] && cp -p "$snap/$f" "$DST/$f"
    done
    return 0
}

report() {
    echo "== VERSION: $(cat "$DST/VERSION" 2>/dev/null || echo none)"
    echo "== $SERVICE: $(systemctl is-active "$SERVICE" || true)"
    local served=""
    for _ in $(seq 1 45); do
        served="$(curl -sf "$CHATBOT_URL/api/version" 2>/dev/null || true)"
        [ -n "$served" ] && break
        sleep 2
    done
    echo "== $CHATBOT_URL/api/version: ${served:-no answer after 90 s}"
}

if [ "${1:-}" = "--rollback" ]; then
    restore "${2:?--rollback SNAPSHOT_DIR}"
    systemctl restart "$SERVICE"
    report
    exit 0
fi

[ -d "$SRC/src" ] && [ -f "$SRC/VERSION" ] || { echo "$SRC has no src/ or VERSION: scripts/deploy.sh first"; exit 1; }
# v79.19.1 (item 0): a symlinked data/ on either side -- rsync would write through the link into another tree (the
# snapshot and the restore likewise): refused before the snapshot, nothing changed
for side in "$SRC" "$DST"; do
    if [ -L "$side/data" ]; then
        echo "refused: $side/data is a symlink ($(readlink "$side/data")): replace it with a directory before promoting"
        exit 1
    fi
done
[ -d "$DST" ] && [ -x "$PY" ] || { echo "no $DST or no $PY"; exit 1; }
[ -f "$EXCLUDE" ] || { echo "no $EXCLUDE"; exit 1; }
old="$(cat "$DST/VERSION" 2>/dev/null || echo unversioned)"
new="$(cat "$SRC/VERSION")"
echo "== promote $new over $old"

# 1. snapshot
snap="$SNAPSHOTS/$(date -u +%Y%m%d-%H%M%S)_${old//[^A-Za-z0-9.+_-]/_}"
mkdir -p "$snap"
for item in $SNAP_ITEMS; do
    if [ -d "$DST/$item" ]; then
        rsync -a --exclude='__pycache__/' "$DST/$item/" "$snap/$item/"
    elif [ -f "$DST/$item" ]; then
        cp -p "$DST/$item" "$snap/$item"
    fi
done
[ -d "$DST/logs" ] && rsync -a "$DST/logs/" "$snap/logs/"         # the logs' old location, before step 5 moves it
echo "== 1. snapshot $snap ($(du -sh "$snap" | cut -f1))"

# 2. rsync, per directory (a failed rsync puts the snapshot back and stops)
sync_dir() {
    local out
    out="$(rsync -a --itemize-changes --exclude-from="$EXCLUDE" "$@")" || {
        restore "$snap"; echo "promote stopped: rsync failed ($*), $DST is back to $old"; exit 1; }
    printf '%s\n' "$out" | grep -v -e '^\.' -e '^$' || true
}
echo "== 2. rsync $SRC -> $DST"
for d in $DIRS_MIRRORED; do
    echo "-- $d/ (mirrored)"
    sync_dir --delete "$SRC/$d/" "$DST/$d/"
done
for d in $DIRS_ADDED; do
    echo "-- $d/ (added and updated, never deleted)"
    sync_dir "$SRC/$d/" "$DST/$d/"
done
cp -p "$SRC/VERSION" "$DST/VERSION"

# 3. flags from .env.v45
echo "== 3. flags from $FLAG_SOURCE"
if [ -f "$FLAG_SOURCE" ]; then
    touch "$DST/.env"
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%$'\r'}"
        [[ "$line" =~ $FLAG_RE ]] || continue
        key="${BASH_REMATCH[1]}"; value="${BASH_REMATCH[2]}"
        [[ "$key" =~ $FLAG_SKIP_RE ]] && continue
        current="$(grep -E "^${key}=" "$DST/.env" | tail -1 | cut -d= -f2- || true)"
        if [ -z "$current" ] && ! grep -qE "^${key}=" "$DST/.env"; then
            printf '%s=%s\n' "$key" "$value" >> "$DST/.env"
            echo "   + $key=$value"
        elif [ "$current" != "$value" ]; then
            sed -i "s|^${key}=.*|${key}=${value}|" "$DST/.env"
            echo "   ~ $key: $current -> $value"
        fi
    done < "$FLAG_SOURCE"
else
    echo "   no $FLAG_SOURCE: .env unchanged"
fi
for key in SESSIONS_FILE SYSTEM_TEMPLATES_CATALOG_PATH ARTICLE_CARDS_FILE CYPHER_LOG_PATH THESIS_LOG_DIR; do
    if grep -qE "^${key}=.*(/data/|/logs)" "$DST/.env"; then
        echo "   warning: .env sets $key to an old location ($(grep -E "^${key}=" "$DST/.env" | tail -1)): the state moves to $DST/state"
    fi
done

# 4. import check
echo "== 4. import check"
if ! (cd "$DST" && "$PY" scripts/import_check.py --env .env); then
    restore "$snap"
    echo "promote stopped: the import check failed, $DST is back to $old, $SERVICE not restarted"
    exit 1
fi

# 5. restart, the runtime state moved once with the service stopped
echo "== 5. restart $SERVICE"
systemctl stop "$SERVICE"
mkdir -p "$DST/state/logs"
move_once() {       # $1 the old path, $2 the new one: moved when only the old exists
    if [ -e "$1" ] && [ ! -e "$2" ]; then
        mv "$1" "$2" && echo "   moved $1 -> $2"
    elif [ -e "$1" ]; then
        echo "   warning: both $1 and $2 exist: $2 is read, $1 left in place"
    fi
}
move_once "$DST/data/sessions.json" "$DST/state/sessions.json"
move_once "$DST/data/tenant_logos" "$DST/state/tenant_logos"
move_once "$DST/data/system_templates" "$DST/state/system_templates"
move_once "$DST/data/article_cards.jsonl" "$DST/state/article_cards.jsonl"
for old_logs in "$DST/data/logs" "$DST/logs"; do      # the logs merged, the old folder kept until it is empty
    if [ -d "$old_logs" ] && [ -n "$(ls -A "$old_logs")" ]; then
        rsync -a --remove-source-files "$old_logs/" "$DST/state/logs/" && echo "   logs moved $old_logs -> $DST/state/logs"
    fi
done
systemctl start "$SERVICE"

# 6. report
report
echo "rollback: bash $SRC/scripts/promote.sh --rollback $snap"
