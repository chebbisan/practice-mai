"""
Benchmark: full 1D pipeline — Pure Python vs NumPy vs C++.
N=16, 1001 angular points, cosine element pattern.

Run with:
    python -m pytest tests/test_benchmark_1d.py -v
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
)

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"


@pytest.fixture(scope="module")
def lib():
    if not _LIB_PATH.exists():
        pytest.skip(f"shared library not found: {_LIB_PATH}")
    return initialize_library(str(_LIB_PATH))


# ---------------------------------------------------------------------------
# Pure Python (no NumPy)
# ---------------------------------------------------------------------------


def _full_pipeline_python():
    import math as _m

    N = 16
    freq = 3e9
    n_theta = 1001
    lam = 3e8 / freq
    d = lam / 2
    k = 2 * _m.pi / lam
    L = d * (N - 1)
    x = [i * d - L / 2 for i in range(N)]
    amps = [1.0] * N
    theta = [(-_m.pi / 2) + i * _m.pi / (n_theta - 1) for i in range(n_theta)]

    af = [0.0] * n_theta
    for i in range(n_theta):
        re, im = 0.0, 0.0
        sin_t = _m.sin(theta[i])
        for j in range(N):
            phase = -k * x[j] * sin_t
            re += amps[j] * _m.cos(phase)
            im += amps[j] * _m.sin(phase)
        af[i] = _m.sqrt(re * re + im * im) / N

    f1 = [abs(_m.cos(t)) for t in theta]
    full = [f1[i] * af[i] for i in range(n_theta)]
    mx = max(full)
    full = [v / mx for v in full]

    integral = 0.0
    for i in range(n_theta - 1):
        dt = theta[i + 1] - theta[i]
        integral += (
            (full[i] ** 2 * _m.cos(theta[i]) + full[i + 1] ** 2 * _m.cos(theta[i + 1]))
            * 0.5
            * dt
        )
    return 2.0 / integral


# ---------------------------------------------------------------------------
# NumPy vectorized
# ---------------------------------------------------------------------------


def _full_pipeline_numpy():
    N = 16
    freq = 3e9
    n_theta = 1001
    lam = 3e8 / freq
    d = lam / 2
    k = 2 * math.pi / lam
    L = d * (N - 1)
    x_arr = np.array([i * d - L / 2 for i in range(N)])
    amps = np.ones(N)
    theta = np.linspace(-math.pi / 2, math.pi / 2, n_theta)

    phase = -k * np.outer(x_arr, np.sin(theta))
    af = np.abs(np.sum(amps[:, np.newaxis] * np.exp(1j * phase), axis=0)) / N

    f1 = np.abs(np.cos(theta))
    full = f1 * af
    full /= full.max()
    return 2.0 / np.trapezoid(full**2 * np.cos(theta), theta)


# ---------------------------------------------------------------------------
# C++ library + NumPy
# ---------------------------------------------------------------------------


def _full_pipeline_cpp(lib):
    N = 16
    freq = 3e9
    n_theta = 1001
    lam = 3e8 / freq
    d = lam / 2
    wave_num = 2 * math.pi / lam
    L = d * (N - 1)
    x_arr = np.array([i * d - L / 2 for i in range(N)])
    theta = np.linspace(-math.pi / 2, math.pi / 2, n_theta)

    f_arr = [complex_t(1.0, 0.0)] * N
    c_f = list_to_c_complex_array(f_arr)
    c_x = list_to_c_double_array(x_arr)
    c_theta = list_to_c_double_array(theta)

    raw = lib.Calculate1DAntennaArray(
        ct.c_int(N), ct.c_int(n_theta), c_f, c_x, c_theta, ct.c_double(wave_num)
    )
    c_af = (ct.c_double * n_theta)()
    lib.ComplexArrayMagnitude(raw, c_af, ct.c_int(n_theta))
    lib.FreeComplexArr(raw)
    af = np.ctypeslib.as_array(c_af)

    f1 = np.abs(np.cos(theta))
    full = f1 * af
    full /= full.max()
    return 2.0 / np.trapezoid(full**2 * np.cos(theta), theta)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_1d_python(benchmark):
    benchmark(_full_pipeline_python)


def test_pipeline_1d_numpy(benchmark):
    benchmark(_full_pipeline_numpy)


def test_pipeline_1d_cpp(benchmark, lib):
    benchmark(_full_pipeline_cpp, lib)
