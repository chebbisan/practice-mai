"""
Benchmark: C library vs pure-Python implementation of Calculate1DAntennaArray.

Run with:
    python -m pytest tests/test_benchmark.py -v --benchmark-compare
"""

import ctypes as ct
import math
import platform
from pathlib import Path

import numpy as np
import pytest

from app.complex import complex_t
from app.util import (
    initialize_library,
    list_to_c_double_array,
    list_to_c_complex_array,
    calculate_delta_x,
    calculate_1d_antenna_array,
)

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"

N = 16
FREQ_0 = 3e9
SPEED_OF_LIGHT = 3e8
WAVE_LENGTH = SPEED_OF_LIGHT / FREQ_0
WAVE_NUM = 2 * math.pi / WAVE_LENGTH

THETA_RANGE = np.linspace(0, 2 * np.pi, 1001)
F_ARR = [complex_t(1, 0) for _ in THETA_RANGE]
DELTA_X = calculate_delta_x(WAVE_LENGTH, math.pi / 6)
L = DELTA_X * (N - 1)
X_ARR = np.array([(i * DELTA_X) - L / 2 for i in range(N)])


@pytest.fixture(scope="module")
def lib():
    if not _LIB_PATH.exists():
        pytest.skip(f"shared library not found: {_LIB_PATH}")
    return initialize_library(str(_LIB_PATH))


@pytest.fixture(scope="module")
def c_arrays():
    return (
        list_to_c_complex_array(F_ARR),
        list_to_c_double_array(X_ARR),
        list_to_c_double_array(THETA_RANGE),
    )


def bench_c_lib(lib, c_arrays):
    c_f, c_x, c_theta = c_arrays
    result = lib.Calculate1DAntennaArray(
        ct.c_int(N),
        ct.c_int(THETA_RANGE.size),
        c_f,
        c_x,
        c_theta,
        ct.c_double(WAVE_NUM),
    )
    lib.FreeComplexArr(result)


def bench_python():
    calculate_1d_antenna_array(N, F_ARR, X_ARR, THETA_RANGE, WAVE_NUM)


def test_benchmark_c_lib(benchmark, lib, c_arrays):
    benchmark(bench_c_lib, lib, c_arrays)


def test_benchmark_python(benchmark):
    benchmark(bench_python)
