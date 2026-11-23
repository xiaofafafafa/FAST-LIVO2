#!/bin/sh
# Run from any directory: explicit failure if standalone build dependencies are absent.
set -eu
cd "$(dirname "$0")/.."
python_bin=${PYTHON_BIN:-python3}
"$python_bin" -m unittest discover -s test -p 'test_gaussian_mapping.py' -v
build=$(mktemp -d)
trap 'rm -rf "$build"' EXIT
cmake -S . -B "$build" -DADAPTIVE_WINDOW_STANDALONE=ON -DBUILD_TESTING=ON -DCMAKE_BUILD_TYPE=Release
cmake --build "$build" -j2
(cd "$build" && ctest --output-on-failure)
