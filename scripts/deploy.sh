#!/usr/bin/env bash
# v77: the Mac -> the test copy (/root/v38). Per directory: src/ scripts/ docs/ tests/ mirrored (rsync --delete: a file
# removed from the repo leaves the copy too), data/ added and updated only (never --delete: the server's own files stay),
# VERSION copied; scripts/deploy_exclude.txt never copied, and so never deleted (env files, venvs, runtime state, logs,
# the held-out set). Then the import check on the server (scripts/import_check.py with .env.v45) and the server's
# VERSION. /opt/chatbot is not touched: scripts/promote.sh, on the server, does that.
#
#   SSH_HOST=1.2.3.4 scripts/deploy.sh                 (SSH_USER root, SSH_PORT 22, SSH_KEY_PATH optional)
# v79.19.1 (item 0): refused (exit 1, nothing synced) when the target's data/ is a symlink.
#   DRY_RUN=1 ...    list what would change, change nothing
#   ALLOW_DIRTY=1 ...  deploy a tree with uncommitted changes (its VERSION then names an older commit)
set -euo pipefail

cd "$(dirname "$0")/.."
: "${SSH_HOST:?set SSH_HOST (the server address)}"
SSH_USER="${SSH_USER:-root}"
SSH_PORT="${SSH_PORT:-22}"
TARGET="${TARGET:-/root/v38}"
DIRS_MIRRORED="src scripts docs tests"
DIRS_ADDED="data"
EXCLUDE="scripts/deploy_exclude.txt"

ssh_opts=(-p "$SSH_PORT" -o ConnectTimeout=15)
[ -z "${SSH_KEY_PATH:-}" ] || ssh_opts+=(-i "$SSH_KEY_PATH")
remote="$SSH_USER@$SSH_HOST"
rsh="ssh ${ssh_opts[*]}"

[ -f VERSION ] || { echo "no VERSION: scripts/release.sh first"; exit 1; }
if [ -n "$(git status --porcelain)" ] && [ -z "${ALLOW_DIRTY:-}" ]; then
    echo "the tree is not clean (ALLOW_DIRTY=1 to deploy it anyway):"; git status --short; exit 1
fi
version="$(head -1 VERSION)"
tagged="$(git tag --points-at HEAD | grep -E '^v[0-9]+\.[0-9]+\.[0-9]+$' || true)"
[ -n "$tagged" ] || echo "warning: HEAD is not a release commit (scripts/release.sh): VERSION says $version"

dry=(); [ -z "${DRY_RUN:-}" ] || dry=(--dry-run)
# one directory: rsync's itemized changes, the unchanged lines left out; a failed rsync stops the deploy
sync_dir() {
    local out
    out="$(rsync -az --itemize-changes ${dry[@]+"${dry[@]}"} "$@")" || { echo "rsync failed: $*"; exit 1; }
    printf '%s\n' "$out" | grep -v -e '^\.' -e '^$' || true
}
echo "== deploy $version -> $remote:$TARGET ${DRY_RUN:+(dry run)}"
ssh "${ssh_opts[@]}" "$remote" "mkdir -p '$TARGET'"
# v79.19.1 (item 0): the target's data/ a symlink -> rsync into it would write through the link into another tree
# (and the mirrored directories' --delete could follow it): refused, nothing synced
if ssh "${ssh_opts[@]}" "$remote" "[ -L '$TARGET/data' ]"; then
    echo "refused: $remote:$TARGET/data is a symlink ($(ssh "${ssh_opts[@]}" "$remote" "readlink '$TARGET/data'")): replace it with a directory before deploying"
    exit 1
fi
for d in $DIRS_MIRRORED; do
    echo "-- $d/ (mirrored)"
    sync_dir --delete --exclude-from="$EXCLUDE" -e "$rsh" "$d/" "$remote:$TARGET/$d/"
done
for d in $DIRS_ADDED; do
    echo "-- $d/ (added and updated, never deleted)"
    sync_dir --exclude-from="$EXCLUDE" -e "$rsh" "$d/" "$remote:$TARGET/$d/"
done
sync_dir -e "$rsh" VERSION "$remote:$TARGET/VERSION"
[ -z "${DRY_RUN:-}" ] || { echo "dry run: nothing changed"; exit 0; }

echo "== import check on $TARGET"
ssh "${ssh_opts[@]}" "$remote" "cd '$TARGET' && if [ -x venv/bin/python ]; then PY=venv/bin/python; else PY=/opt/chatbot/venv/bin/python; fi && \$PY scripts/import_check.py --env .env.v45"
echo "== the server's VERSION: $(ssh "${ssh_opts[@]}" "$remote" "cat '$TARGET/VERSION'")"
echo "next, on the server: scripts/serve_test.sh for a hand test; then bash $TARGET/scripts/promote.sh"
