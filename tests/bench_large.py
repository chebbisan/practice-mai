"""
One-shot benchmark on LARGE inputs.
1D: N=4096, n_theta=20001
2D: 64x64=4096 elements, 361x361 grid
3D: 16x16x8=2048 elements, 181x181 grid
Pure-Python skipped for 2D/3D (would take hours).
"""

import ctypes as ct
import math
import platform
import time
from pathlib import Path

import numpy as np

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from app.complex import complex_t
from app.util import initialize_library, list_to_c_double_array, list_to_c_complex_array

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
LIB = initialize_library(str(ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"))

REPEATS = 3


def bench(fn, repeats=REPEATS):
    fn()  # warmup
    times = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        times.append(time.perf_counter() - t0)
    return min(times)


# ============================================================================
# 1D — N=4096, n_theta=20001
# ============================================================================
N1, NT1 = 4096, 20001


def _1d_python():
    import math as _m
    lam = 0.1
    d = lam / 2
    k = 2 * _m.pi / lam
    L = d * (N1 - 1)
    x = [i * d - L / 2 for i in range(N1)]
    amps = [1.0] * N1
    theta = [(-_m.pi / 2) + i * _m.pi / (NT1 - 1) for i in range(NT1)]
    af = [0.0] * NT1
    for i in range(NT1):
        re, im = 0.0, 0.0
        sin_t = _m.sin(theta[i])
        for j in range(N1):
            phase = -k * x[j] * sin_t
            re += amps[j] * _m.cos(phase)
            im += amps[j] * _m.sin(phase)
        af[i] = _m.sqrt(re * re + im * im) / N1
    f1 = [abs(_m.cos(t)) for t in theta]
    full = [f1[i] * af[i] for i in range(NT1)]
    mx = max(full)
    full = [v / mx for v in full]
    integral = 0.0
    for i in range(NT1 - 1):
        dt = theta[i + 1] - theta[i]
        integral += (full[i] ** 2 * _m.cos(theta[i]) + full[i + 1] ** 2 * _m.cos(theta[i + 1])) * 0.5 * dt
    return 2.0 / integral


def _1d_numpy():
    lam = 0.1
    d = lam / 2
    k = 2 * math.pi / lam
    L = d * (N1 - 1)
    x_arr = np.arange(N1) * d - L / 2
    amps = np.ones(N1)
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT1)
    phase = -k * np.outer(x_arr, np.sin(theta))
    af = np.abs(np.sum(amps[:, np.newaxis] * np.exp(1j * phase), axis=0)) / N1
    f1 = np.abs(np.cos(theta))
    full = f1 * af
    full /= full.max()
    return 2.0 / np.trapezoid(full**2 * np.cos(theta), theta)


def _1d_cpp_hybrid():
    lam = 0.1
    d = lam / 2
    wave_num = 2 * math.pi / lam
    L = d * (N1 - 1)
    x_arr = np.arange(N1) * d - L / 2
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT1)
    f_arr = [complex_t(1.0, 0.0)] * N1
    raw = LIB.Calculate1DAntennaArray(
        ct.c_int(N1), ct.c_int(NT1),
        list_to_c_complex_array(f_arr),
        list_to_c_double_array(x_arr),
        list_to_c_double_array(theta),
        ct.c_double(wave_num),
    )
    c_af = (ct.c_double * NT1)()
    LIB.ComplexArrayMagnitude(raw, c_af, ct.c_int(NT1))
    LIB.FreeComplexArr(raw)
    af = np.ctypeslib.as_array(c_af)
    f1 = np.abs(np.cos(theta))
    full = f1 * af
    full /= full.max()
    return 2.0 / np.trapezoid(full**2 * np.cos(theta), theta)


def _1d_pure_cpp():
    lam = 0.1
    d = lam / 2
    wave_num = 2 * math.pi / lam
    L = d * (N1 - 1)
    x_arr = np.arange(N1) * d - L / 2
    amps = np.ones(N1)
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT1)
    return LIB.FullPipeline1D(
        ct.c_int(N1), ct.c_int(NT1),
        list_to_c_double_array(x_arr),
        list_to_c_double_array(amps),
        list_to_c_double_array(theta),
        ct.c_double(wave_num),
    )


# ============================================================================
# 2D — 64x64 = 4096 elements, 361x361 grid
# ============================================================================
NX2, NY2 = 64, 64
N2 = NX2 * NY2
NT2, NP2 = 361, 361


def _make_grid_2d():
    lam = 0.1
    d = lam / 2
    xs = np.array([(ix - (NX2 - 1) / 2) * d for ix in range(NX2) for _ in range(NY2)])
    ys = np.array([(iy - (NY2 - 1) / 2) * d for _ in range(NX2) for iy in range(NY2)])
    return xs, ys, 2 * math.pi / lam


def _2d_numpy():
    xs, ys, k = _make_grid_2d()
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT2)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP2)
    st_cp = np.outer(np.sin(theta), np.cos(phi))
    st_sp = np.outer(np.sin(theta), np.sin(phi))
    af_c = np.zeros((NT2, NP2), dtype=complex)
    for n in range(N2):
        af_c += np.exp(-1j * k * (xs[n] * st_cp + ys[n] * st_sp))
    af = np.abs(af_c) / N2
    f1 = np.abs(np.cos(theta))
    full = f1[:, None] * af
    full /= full.max()
    inner = np.trapezoid(full**2 * np.cos(theta)[:, None], theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


def _2d_cpp_hybrid():
    xs, ys, k = _make_grid_2d()
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT2)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP2)
    f_arr = [complex_t(1.0, 0.0)] * N2
    raw = LIB.Calculate2DAntennaArray(
        ct.c_int(N2), ct.c_int(NT2), ct.c_int(NP2),
        list_to_c_complex_array(f_arr),
        list_to_c_double_array(xs), list_to_c_double_array(ys),
        list_to_c_double_array(theta), list_to_c_double_array(phi),
        ct.c_double(k),
    )
    tot = NT2 * NP2
    c_af = (ct.c_double * tot)()
    LIB.ComplexArrayMagnitude(raw, c_af, ct.c_int(tot))
    LIB.FreeComplexArr(raw)
    af = np.ctypeslib.as_array(c_af).reshape(NT2, NP2)
    f1 = np.abs(np.cos(theta))
    full = f1[:, None] * af
    full /= full.max()
    inner = np.trapezoid(full**2 * np.cos(theta)[:, None], theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


def _2d_pure_cpp():
    xs, ys, k = _make_grid_2d()
    amps = np.ones(N2)
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT2)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP2)
    return LIB.FullPipeline2D(
        ct.c_int(N2), ct.c_int(NT2), ct.c_int(NP2),
        list_to_c_double_array(xs), list_to_c_double_array(ys),
        list_to_c_double_array(amps),
        list_to_c_double_array(theta), list_to_c_double_array(phi),
        ct.c_double(k),
    )


# ============================================================================
# 3D — 16x16x8 = 2048 elements, 181x181 grid
# ============================================================================
NX3, NY3, NZ3 = 16, 16, 8
N3 = NX3 * NY3 * NZ3
NT3, NP3 = 181, 181


def _make_grid_3d():
    lam = 0.1
    d = lam / 2
    xs, ys, zs = [], [], []
    for ix in range(NX3):
        for iy in range(NY3):
            for iz in range(NZ3):
                xs.append((ix - (NX3 - 1) / 2) * d)
                ys.append((iy - (NY3 - 1) / 2) * d)
                zs.append((iz - (NZ3 - 1) / 2) * d)
    return np.array(xs), np.array(ys), np.array(zs), 2 * math.pi / lam


def _3d_numpy():
    xs, ys, zs, k = _make_grid_3d()
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT3)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP3)
    st_cp = np.outer(np.sin(theta), np.cos(phi))
    st_sp = np.outer(np.sin(theta), np.sin(phi))
    ct_2d = np.cos(theta)[:, None] * np.ones((1, NP3))
    af_c = np.zeros((NT3, NP3), dtype=complex)
    for n in range(N3):
        af_c += np.exp(-1j * k * (xs[n] * st_cp + ys[n] * st_sp + zs[n] * ct_2d))
    af = np.abs(af_c) / N3
    f1 = np.abs(np.cos(theta))
    full = f1[:, None] * af
    full /= full.max()
    inner = np.trapezoid(full**2 * np.cos(theta)[:, None], theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


def _3d_cpp_hybrid():
    xs, ys, zs, k = _make_grid_3d()
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT3)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP3)
    f_arr = [complex_t(1.0, 0.0)] * N3
    raw = LIB.Calculate3DAntennaArray(
        ct.c_int(N3), ct.c_int(NT3), ct.c_int(NP3),
        list_to_c_complex_array(f_arr),
        list_to_c_double_array(xs), list_to_c_double_array(ys), list_to_c_double_array(zs),
        list_to_c_double_array(theta), list_to_c_double_array(phi),
        ct.c_double(k),
    )
    tot = NT3 * NP3
    c_af = (ct.c_double * tot)()
    LIB.ComplexArrayMagnitude(raw, c_af, ct.c_int(tot))
    LIB.FreeComplexArr(raw)
    af = np.ctypeslib.as_array(c_af).reshape(NT3, NP3)
    f1 = np.abs(np.cos(theta))
    full = f1[:, None] * af
    full /= full.max()
    inner = np.trapezoid(full**2 * np.cos(theta)[:, None], theta, axis=0)
    return 4 * math.pi / np.trapezoid(inner, phi)


def _3d_pure_cpp():
    xs, ys, zs, k = _make_grid_3d()
    amps = np.ones(N3)
    theta = np.linspace(-math.pi / 2, math.pi / 2, NT3)
    phi = np.linspace(-math.pi / 2, math.pi / 2, NP3)
    return LIB.FullPipeline3D(
        ct.c_int(N3), ct.c_int(NT3), ct.c_int(NP3),
        list_to_c_double_array(xs), list_to_c_double_array(ys), list_to_c_double_array(zs),
        list_to_c_double_array(amps),
        list_to_c_double_array(theta), list_to_c_double_array(phi),
        ct.c_double(k),
    )


# ============================================================================
# Run
# ============================================================================
def fmt(t):
    if t >= 1.0:
        return f"{t:.2f} s"
    if t >= 1e-3:
        return f"{t * 1e3:.1f} ms"
    return f"{t * 1e6:.0f} us"


print(f"\n=== 1D : N={N1}, n_theta={NT1} ===")
print(f"  Pure Python     : RUNNING (slow)...", flush=True)
t_py1 = bench(_1d_python, repeats=1)
print(f"  Pure Python     : {fmt(t_py1)}")
t_np1 = bench(_1d_numpy)
print(f"  NumPy           : {fmt(t_np1)}")
t_hy1 = bench(_1d_cpp_hybrid)
print(f"  C++ hybrid      : {fmt(t_hy1)}")
t_pc1 = bench(_1d_pure_cpp)
print(f"  Pure C++        : {fmt(t_pc1)}")

print(f"\n=== 2D : {NX2}x{NY2}={N2} elem, grid {NT2}x{NP2} ===")
t_np2 = bench(_2d_numpy, repeats=2)
print(f"  NumPy           : {fmt(t_np2)}")
t_hy2 = bench(_2d_cpp_hybrid, repeats=2)
print(f"  C++ hybrid      : {fmt(t_hy2)}")
t_pc2 = bench(_2d_pure_cpp, repeats=2)
print(f"  Pure C++        : {fmt(t_pc2)}")

print(f"\n=== 3D : {NX3}x{NY3}x{NZ3}={N3} elem, grid {NT3}x{NP3} ===")
t_np3 = bench(_3d_numpy, repeats=2)
print(f"  NumPy           : {fmt(t_np3)}")
t_hy3 = bench(_3d_cpp_hybrid, repeats=2)
print(f"  C++ hybrid      : {fmt(t_hy3)}")
t_pc3 = bench(_3d_pure_cpp, repeats=2)
print(f"  Pure C++        : {fmt(t_pc3)}")

print("\n=== SUMMARY (min of repeats) ===")
print(f"{'Case':<35} {'Python':>12} {'NumPy':>12} {'C++ hybrid':>12} {'Pure C++':>12}")
print(f"{'-' * 87}")
print(f"{'1D N=' + str(N1) + ', n_theta=' + str(NT1):<35} {fmt(t_py1):>12} {fmt(t_np1):>12} {fmt(t_hy1):>12} {fmt(t_pc1):>12}")
print(f"{'2D ' + str(NX2) + 'x' + str(NY2) + ' grid ' + str(NT2) + 'x' + str(NP2):<35} {'-':>12} {fmt(t_np2):>12} {fmt(t_hy2):>12} {fmt(t_pc2):>12}")
print(f"{'3D ' + str(NX3) + 'x' + str(NY3) + 'x' + str(NZ3) + ' grid ' + str(NT3) + 'x' + str(NP3):<35} {'-':>12} {fmt(t_np3):>12} {fmt(t_hy3):>12} {fmt(t_pc3):>12}")
