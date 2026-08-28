#!/usr/bin/env bash

set -euo pipefail

WHEEL=$(find dist -maxdepth 1 -name "*.whl" -print -quit)

if [[ -z "${WHEEL}" ]]; then
    echo "No wheel found in dist/"
    echo "Run ./scripts/build_wheel.sh first."
    exit 1
fi

echo "Testing wheel: ${WHEEL}"

TEST_ENV=$(mktemp -d)

cleanup() {
    rm -rf "${TEST_ENV}"
}

trap cleanup EXIT

echo "Test environment: ${TEST_ENV}"

uv venv "${TEST_ENV}/.venv"

uv pip install \
    --python "${TEST_ENV}/.venv/bin/python" \
    "${WHEEL}" \
    "numpy>=1.24" \
    "pytest>=8"

uv run \
    --python "${TEST_ENV}/.venv/bin/python" \
    pytest tests/python -v

echo
echo "Wheel test passed."