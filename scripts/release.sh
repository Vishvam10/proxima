#!/usr/bin/env bash
set -euo pipefail

VERSION="${1:-}"

if [[ -z "$VERSION" ]]; then
echo "Usage: ./scripts/release.sh <version>"
echo "Example: ./scripts/release.sh 0.1.0"
exit 1
fi

if [[ ! "$VERSION" =~ ^[0-9]+.[0-9]+.[0-9]+$ ]]; then
echo "Version must have the form X.Y.Z"
exit 1
fi

TAG="v${VERSION}"

if git rev-parse "$TAG" >/dev/null 2>&1; then
echo "Tag $TAG already exists"
exit 1
fi

echo "Preparing release $TAG"

git diff --exit-code

echo
echo "Current commit:"
git log -1 --oneline

echo
read -r -p "Create release $TAG? [y/N] " answer

if [[ "$answer" != "y" && "$answer" != "Y" ]]; then
echo "Aborted."
exit 1
fi

git tag -a "$TAG" -m "Release $TAG"

git push origin "$TAG"

echo
echo "Released tag: $TAG"
echo "GitHub Actions will build and publish the release."
