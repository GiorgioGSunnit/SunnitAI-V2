#!/usr/bin/env bash
# v77: cut a release -- the VERSION file at the repository root, «<semver>+<git short sha>», committed and tagged
# v<semver>. The sha is the code commit the release names (HEAD when this runs); the release commit itself only writes
# VERSION. Run it on a clean tree, before scripts/deploy.sh: one release per deploy.
#
#   scripts/release.sh 77.0.0          an explicit version
#   scripts/release.sh [patch]         the next patch of VERSION's semver (default)
#   scripts/release.sh minor | major
#
# Nothing is pushed: it prints the push command. RELEASE_NOTE="..." adds a paragraph to the commit message.
set -euo pipefail

cd "$(dirname "$0")/.."
[ -z "$(git status --porcelain)" ] || { echo "the tree is not clean: commit or stash first"; git status --short; exit 1; }

current="0.0.0"
if [ -f VERSION ]; then
    current="$(head -1 VERSION | sed 's/+.*//')"
fi
[[ "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]] || { echo "VERSION's semver «$current» is not X.Y.Z"; exit 1; }
major="${BASH_REMATCH[1]}"; minor="${BASH_REMATCH[2]}"; patch="${BASH_REMATCH[3]}"

case "${1:-patch}" in
    patch) next="$major.$minor.$((patch + 1))" ;;
    minor) next="$major.$((minor + 1)).0" ;;
    major) next="$((major + 1)).0.0" ;;
    *)
        next="$1"
        [[ "$next" =~ ^[0-9]+\.[0-9]+\.[0-9]+$ ]] || { echo "usage: scripts/release.sh [X.Y.Z | patch | minor | major]"; exit 1; } ;;
esac
tag="v$next"
if git rev-parse -q --verify "refs/tags/$tag" > /dev/null; then echo "tag $tag exists"; exit 1; fi

sha="$(git rev-parse --short HEAD)"
version="$next+$sha"
echo "$version" > VERSION
git add VERSION
message="release $next ($sha)"
[ -z "${RELEASE_NOTE:-}" ] || message="$message"$'\n\n'"$RELEASE_NOTE"
git commit -q -m "$message"
git tag -a "$tag" -m "SunnitAI $version"
echo "released $version: commit $(git rev-parse --short HEAD), tag $tag (was $current)"
echo "push: git push origin $(git rev-parse --abbrev-ref HEAD) $tag    then: scripts/deploy.sh"
