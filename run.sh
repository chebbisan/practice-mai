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
    1d)    python3 "$ROOT/app/calc_1d.py" ;;
    2d)    python3 "$ROOT/app/calc_2d.py" ;;
    app)   python3 "$ROOT/app/gui.py" ;;
    bench) python3 -m pytest "$ROOT/tests/test_benchmark.py" "$ROOT/tests/test_benchmark_1d.py" "$ROOT/tests/test_benchmark_2d.py" -v ;;
    test)  python3 -m pytest "$ROOT/tests/test_antenna_array.py" -v ;;
    *)
        echo "Usage: $0 {1d|2d|app|bench|test}"
        echo ""
        echo "  1d     Run 1D antenna array (calc_1d.py)"
        echo "  2d     Run 2D antenna array from CSV (calc_2d.py)"
        echo "  app    Run PyQt6 GUI (gui.py)"
        echo "  bench  Run benchmark suite"
        echo "  test   Run test suite"
        exit 0
        ;;
esac
