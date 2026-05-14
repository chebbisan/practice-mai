"""
Эксперимент: на каком количестве элементов N C++ Calculate2DAntennaArray
обгоняет NumPy с учётом ctypes-маршаллинга.

Сетка углов фиксирована (101×101), варьируется N.

Запуск:
    python -m pytest tests/test_benchmark_crossover_2d.py -v \\
        --benchmark-group-by=param:N \\
        --benchmark-columns=median,mean,rounds
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
    list_to_c_complex_array,
    list_to_c_double_array,
)

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"

N_THETA, N_PHI = 101, 101

# Шаг ~×3: от очень малых решёток до очень больших
N_SIZES = [16, 49, 144, 400, 900, 2025, 4900, 10000]


@pytest.fixture(scope="module")
def lib():
    if not _LIB_PATH.exists():
        pytest.skip(f"shared library not found: {_LIB_PATH}")
    return initialize_library(str(_LIB_PATH))


def _make_grid(N):
    """Квадратная решётка nx*ny ≈ N с шагом λ/2."""
    lam = 3e8 / 3e9
    d = lam / 2
    nx = int(math.ceil(math.sqrt(N)))
    ny = int(math.ceil(N / nx))
    xs, ys = [], []
    for ix in range(nx):
        for iy in range(ny):
            xs.append((ix - (nx - 1) / 2) * d)
            ys.append((iy - (ny - 1) / 2) * d)
    return np.array(xs[:N]), np.array(ys[:N]), 2 * math.pi / lam


def _numpy_af(x, y, theta, phi, k):
    N = x.size
    st_cp = np.outer(np.sin(theta), np.cos(phi))
    st_sp = np.outer(np.sin(theta), np.sin(phi))
    af = np.zeros((theta.size, phi.size), dtype=complex)
    for n in range(N):
        af += np.exp(-1j * k * (x[n] * st_cp + y[n] * st_sp))
    return np.abs(af) / N


def _cpp_af(lib, x, y, theta, phi, k):
    N = x.size
    n_theta, n_phi = theta.size, phi.size
    f_arr = [complex_t(1.0, 0.0)] * N
    raw = lib.Calculate2DAntennaArray(
        ct.c_int(N),
        ct.c_int(n_theta),
        ct.c_int(n_phi),
        list_to_c_complex_array(f_arr),
        list_to_c_double_array(x),
        list_to_c_double_array(y),
        list_to_c_double_array(theta),
        list_to_c_double_array(phi),
        ct.c_double(k),
    )
    total = n_theta * n_phi
    out = (ct.c_double * total)()
    lib.ComplexArrayMagnitude(raw, out, ct.c_int(total))
    lib.FreeComplexArr(raw)
    return np.ctypeslib.as_array(out).reshape(n_theta, n_phi)


def test_correctness(lib):
    """NumPy и C++ дают совпадающий АФР с точностью до 1e-9."""
    x, y, k = _make_grid(64)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)
    a = _numpy_af(x, y, theta, phi, k)
    b = _cpp_af(lib, x, y, theta, phi, k)
    assert np.allclose(a, b, atol=1e-9)


@pytest.mark.parametrize("N", N_SIZES)
def test_numpy_2d_af(benchmark, N):
    x, y, k = _make_grid(N)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)
    benchmark(_numpy_af, x, y, theta, phi, k)


@pytest.mark.parametrize("N", N_SIZES)
def test_cpp_2d_af(benchmark, lib, N):
    x, y, k = _make_grid(N)
    theta = np.linspace(-math.pi / 2, math.pi / 2, N_THETA)
    phi = np.linspace(-math.pi / 2, math.pi / 2, N_PHI)
    benchmark(_cpp_af, lib, x, y, theta, phi, k)
