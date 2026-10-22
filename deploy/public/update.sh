#!/bin/sh
# Update a public LUMOS instance (docs/web/hosting.md): pull the new version and rebuild the image with it.
# The image is built without the git history, so its version is computed here and passed as a build argument.
#
#   sh deploy/public/update.sh            # the latest commit of main
#   sh deploy/public/update.sh v1.0.0     # a release
set -eu
cd "$(dirname "$0")/../.."
if [ "$#" -gt 0 ]; then
    git fetch --tags
    git checkout "$1"
else
    git checkout -q main  # after a release tag, git is on a detached HEAD
    git pull --ff-only
fi
LUMOS_VERSION="$(python3 tools/release_version.py)"
export LUMOS_VERSION
echo "building LUMOS $LUMOS_VERSION"
docker compose -f deploy/public/compose.yaml up -d --build
