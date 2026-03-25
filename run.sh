#!/usr/bin/env bash
set -e

ROOT="$(cd "$(dirname "$0")" && pwd)"
BUILD_DIR="$ROOT/build"

# Configure if not yet done
if [ ! -f "$BUILD_DIR/CMakeCache.txt" ]; then
    cmake -S "$ROOT" -B "$BUILD_DIR"
fi

# Build shared library (only recompiles if sources changed)
cmake --build "$BUILD_DIR"

case "$1" in
    1d)    python3 "$ROOT/app/main.py" ;;
    app)   python3 "$ROOT/app/app.py" ;;
    nb)    jupyter notebook "$ROOT/notebooks/antenna_array.ipynb" ;;
    bench) python3 -m pytest "$ROOT/tests/test_benchmark.py" ;;
    test)  python3 -m pytest "$ROOT/tests/test_antenna_array.py" -v ;;
    *)
        echo "Usage: $0 {1d|app|nb|bench|test}"
        echo ""
        echo "  1d     Run 1D antenna array (main.py)"
        echo "  app    Run PyQt6 GUI (app.py)"
        echo "  nb     Open main notebook (notebooks/antenna_array.ipynb)"
        echo "  bench  Run benchmark suite"
        echo "  test   Run test suite"
        exit 0
        ;;
esac
