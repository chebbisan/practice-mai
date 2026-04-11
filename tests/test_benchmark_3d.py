"""
Benchmark: full 3D pipeline — Pure Python vs NumPy vs C++ vs Pure C++.
4x4x2 cuboid array (32 elements), 101x101 angular points, cosine element pattern.

Run with:
    python -m pytest tests/test_benchmark_3d.py -v
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

NX, NY, NZ = 4, 4, 2
N_TOTAL = NX * NY * NZ
N_THETA, N_PHI = 101, 101


@pytest.fixture(scope="module")
def lib():
    if not _LIB_PATH.exists():
        pytest.skip(f"shared library not found: {_LIB_PATH}")
    return initialize_library(str(_LIB_PATH))


def _make_grid():
    freq = 3e9
    lam = 3e8 / freq
    d = lam / 2
    xs, ys, zs = [], [], []
    for ix in range(NX):
        for iy in range(NY):
            for iz in range(NZ):
                xs.append((ix - (NX - 1) / 2) * d)
                ys.append((iy - (NY - 1) / 2) * d)
                zs.append((iz - (NZ - 1) / 2) * d)
    return xs, ys, zs, 2 * math.pi / lam


# ---------------------------------------------------------------------------
# Pure Python (no NumPy)
# ---------------------------------------------------------------------------


def _full_pipeline_python():
    import math as _m

    xs, ys, zs, k = _make_grid()
    amps = [1.0] * N_TOTAL
    theta = [(-_m.pi / 2) + i * _m.pi / (N_THETA - 1) for i in range(N_THETA)]
    phi = [(-_m.pi / 2) + i * _m.pi / (N_PHI - 1) for i in range(N_PHI)]

    af = [0.0] * (N_THETA * N_PHI)
    for i in range(N_THETA):
        sin_t = _m.sin(theta[i])
        cos_t = _m.cos(theta[i])
        for j in range(N_PHI):
            cos_p = _m.cos(phi[j])
            sin_p = _m.sin(phi[j])
            re, im = 0.0, 0.0
            for n in range(N_TOTAL):
                phase = -k * (
                    xs[n] * sin_t * cos_p + ys[n] * sin_t * sin_p + zs[n] * cos_t
                )
                re += amps[n] * _m.cos(phase)
                im += amps[n] * _m.sin(phase)
            af[i * N_PHI + j] = _m.sqrt(re * re + im * im) / N_TOTAL

    f1 = [abs(_m.cos(t)) for t in theta]
    mx = 0.0
    full = [0.0] * (N_THETA * N_PHI)
    for i in range(N_THETA):
        for j in range(N_PHI):
            v = f1[i] * af[i * N_PHI + j]
            full[i * N_PHI + j] = v
            if v > mx:
                mx = v
    full = [v / mx for v in full]

    inner = [0.0] * N_PHI
    for j in range(N_PHI):
        s = 0.0
        for i in range(N_THETA - 1):
            dt = theta[i + 1] - theta[i]
            s += (
                (
                    full[i * N_PHI + j] ** 2 * _m.cos(theta[i])
                    + full[(i + 1) * N_PHI + j] ** 2 * _m.cos(theta[i + 1])
                )
                * 0.5
                * dt
            )
        inner[j] = s
    integral = 0.0
    for j in range(N_PHI - 1):
        dp = phi[j + 1] - phi[j]
        integral += (inner[j] + inner[j + 1]) * 0.5 * dp
    return 4.0 * _m.pi / integral


# ---------------------------------------------------------------------------
# NumPy vectorized
# ---------------------------------------------------------------------------


def _full_pipeline_numpy():
    xs, ys, zs, k = _make_grid()
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    z_arr = np.array(zs)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)

    st_cp = np.outer(np.sin(theta), np.cos(phi))
    st_sp = np.outer(np.sin(theta), np.sin(phi))
    cos_theta_2d = np.cos(theta)[:, np.newaxis] * np.ones((1, N_PHI))
    af_complex = np.zeros((N_THETA, N_PHI), dtype=complex)
    for n in range(N_TOTAL):
        af_complex += np.exp(
            -1j * k * (x_arr[n] * st_cp + y_arr[n] * st_sp + z_arr[n] * cos_theta_2d)
        )
    af_2d = np.abs(af_complex) / N_TOTAL

    f1 = np.abs(np.cos(theta))
    full = f1[:, np.newaxis] * af_2d
    full /= full.max()

    integrand = full**2 * np.cos(theta)[:, np.newaxis]
    inner = np.trapezoid(integrand, theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


# ---------------------------------------------------------------------------
# C++ library + NumPy
# ---------------------------------------------------------------------------


def _full_pipeline_cpp(lib):
    xs, ys, zs, k = _make_grid()
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    z_arr = np.array(zs)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)

    f_arr = [complex_t(1.0, 0.0)] * N_TOTAL
    raw = lib.Calculate3DAntennaArray(
        ct.c_int(N_TOTAL),
        ct.c_int(N_THETA),
        ct.c_int(N_PHI),
        list_to_c_complex_array(f_arr),
        list_to_c_double_array(x_arr),
        list_to_c_double_array(y_arr),
        list_to_c_double_array(z_arr),
        list_to_c_double_array(theta),
        list_to_c_double_array(phi),
        ct.c_double(k),
    )
    total = N_THETA * N_PHI
    c_af = (ct.c_double * total)()
    lib.ComplexArrayMagnitude(raw, c_af, ct.c_int(total))
    lib.FreeComplexArr(raw)
    af_2d = np.ctypeslib.as_array(c_af).reshape(N_THETA, N_PHI)

    f1 = np.abs(np.cos(theta))
    full = f1[:, np.newaxis] * af_2d
    full /= full.max()

    integrand = full**2 * np.cos(theta)[:, np.newaxis]
    inner = np.trapezoid(integrand, theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


# ---------------------------------------------------------------------------
# Pure C++ (single ctypes call)
# ---------------------------------------------------------------------------


def _full_pipeline_pure_cpp(lib):
    xs, ys, zs, k = _make_grid()
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    z_arr = np.array(zs)
    amps = np.ones(N_TOTAL)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)

    c_x = list_to_c_double_array(x_arr)
    c_y = list_to_c_double_array(y_arr)
    c_z = list_to_c_double_array(z_arr)
    c_amps = list_to_c_double_array(amps)
    c_theta = list_to_c_double_array(theta)
    c_phi = list_to_c_double_array(phi)

    return lib.FullPipeline3D(
        ct.c_int(N_TOTAL),
        ct.c_int(N_THETA),
        ct.c_int(N_PHI),
        c_x,
        c_y,
        c_z,
        c_amps,
        c_theta,
        c_phi,
        ct.c_double(k),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_pipeline_3d_python(benchmark):
    benchmark.pedantic(_full_pipeline_python, rounds=3, warmup_rounds=1)


def test_pipeline_3d_numpy(benchmark):
    benchmark(_full_pipeline_numpy)


def test_pipeline_3d_cpp(benchmark, lib):
    benchmark(_full_pipeline_cpp, lib)


def test_pipeline_3d_pure_cpp(benchmark, lib):
    benchmark(_full_pipeline_pure_cpp, lib)
