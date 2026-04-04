"""
Benchmark: C library vs pure-Python for individual operations.

Run with:
    python -m pytest tests/test_benchmark.py -v
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


# ---------------------------------------------------------------------------
# 1D Array Factor: C++ vs Python
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Magnitude extraction: Python loop vs C++
# ---------------------------------------------------------------------------

N_MAG = 64
N_THETA_MAG = 10001
_WAVE_NUM_MAG = 2 * math.pi / WAVE_LENGTH
_D_MAG = WAVE_LENGTH / 2
_X_MAG = np.array([(i - (N_MAG - 1) / 2) * _D_MAG for i in range(N_MAG)])
_THETA_MAG = np.linspace(-math.pi / 2, math.pi / 2, N_THETA_MAG)
_F_MAG = [complex_t(1.0, 0.0)] * N_MAG


@pytest.fixture(scope="module")
def raw_complex_1d(lib):
    c_f = list_to_c_complex_array(_F_MAG)
    c_x = list_to_c_double_array(_X_MAG)
    c_theta = list_to_c_double_array(_THETA_MAG)
    raw = lib.Calculate1DAntennaArray(
        ct.c_int(N_MAG),
        ct.c_int(N_THETA_MAG),
        c_f,
        c_x,
        c_theta,
        ct.c_double(_WAVE_NUM_MAG),
    )
    yield raw
    lib.FreeComplexArr(raw)


def test_benchmark_magnitude_python_loop(benchmark, raw_complex_1d):
    benchmark(
        lambda: np.array(
            [
                abs(complex(raw_complex_1d[i].real, raw_complex_1d[i].imag))
                for i in range(N_THETA_MAG)
            ]
        )
    )


def test_benchmark_magnitude_cpp(benchmark, lib, raw_complex_1d):
    def _run():
        c_out = (ct.c_double * N_THETA_MAG)()
        lib.ComplexArrayMagnitude(raw_complex_1d, c_out, ct.c_int(N_THETA_MAG))
        return np.ctypeslib.as_array(c_out)

    benchmark(_run)


# ---------------------------------------------------------------------------
# Normalization: NumPy vs C++
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def af_array(lib, raw_complex_1d):
    c_out = (ct.c_double * N_THETA_MAG)()
    lib.ComplexArrayMagnitude(raw_complex_1d, c_out, ct.c_int(N_THETA_MAG))
    return np.ctypeslib.as_array(c_out).copy()


def test_benchmark_normalize_numpy(benchmark, af_array):
    benchmark(lambda: af_array.copy() / af_array.max())


def test_benchmark_normalize_cpp(benchmark, lib, af_array):
    def _run():
        c_arr = (ct.c_double * len(af_array))(*af_array)
        lib.NormalizeArray(c_arr, ct.c_int(len(af_array)))

    benchmark(_run)


# ---------------------------------------------------------------------------
# Directivity 1D: NumPy vs C++
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def normalized_pattern(af_array):
    return af_array / af_array.max()


def test_benchmark_directivity_numpy(benchmark, normalized_pattern):
    benchmark(
        lambda: (
            2.0 / np.trapezoid(normalized_pattern**2 * np.cos(_THETA_MAG), _THETA_MAG)
        )
    )


def test_benchmark_directivity_cpp(benchmark, lib, normalized_pattern):
    def _run():
        c_pat = (ct.c_double * len(normalized_pattern))(*normalized_pattern)
        c_theta = (ct.c_double * len(_THETA_MAG))(*_THETA_MAG)
        return lib.CalculateDirectivity1D(c_pat, c_theta, ct.c_int(len(_THETA_MAG)))

    benchmark(_run)
