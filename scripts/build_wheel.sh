#!/usr/bin/env bash

set -euo pipefail

echo "Building Proxima packages..."

rm -rf dist

uv build

echo
echo "Built packages:"
ls -lh dist/