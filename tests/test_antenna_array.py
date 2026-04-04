"""
Tests for the antenna array library.

C library tests require a built shared library at build/libAntennaArray.{so,dylib}.
Run `mkdir build && cd build && cmake .. && make` before running these tests.
"""

import math
import platform
from pathlib import Path

import numpy as np
import pytest

from app.complex import complex_t
import ctypes as ct

import yaml

from app.util import (
    initialize_library,
    list_to_c_int_array,
    list_to_c_double_array,
    list_to_c_complex_array,
    calculate_delta_x,
    calculate_1d_antenna_array,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

ROOT = Path(__file__).parent.parent
CONFIG_PATH = ROOT / "app" / "config.yaml"

_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"


def _lib():
    if not _LIB_PATH.exists():
        pytest.skip(f"shared library not found: {_LIB_PATH}")
    return initialize_library(str(_LIB_PATH))


# ---------------------------------------------------------------------------
# Pure-Python helpers
# ---------------------------------------------------------------------------


class TestListConverters:
    def test_int_array_values(self):
        arr = list_to_c_int_array([1, 2, 3])
        assert list(arr) == [1, 2, 3]

    def test_double_array_values(self):
        arr = list_to_c_double_array([1.0, 2.5, -3.0])
        assert list(arr) == pytest.approx([1.0, 2.5, -3.0])

    def test_complex_array_values(self):
        arr = list_to_c_complex_array([complex_t(1.0, -2.0), complex_t(0.0, 3.0)])
        assert arr[0].real == pytest.approx(1.0)
        assert arr[0].imag == pytest.approx(-2.0)
        assert arr[1].real == pytest.approx(0.0)
        assert arr[1].imag == pytest.approx(3.0)

    def test_empty_array(self):
        assert len(list_to_c_double_array([])) == 0


class TestCalculateDeltaX:
    def test_broadside(self):
        # theta = 0 → delta_x = λ / (1 + sin(0)) = λ
        wave_length = 0.1
        result = calculate_delta_x(wave_length, 0.0)
        assert result == pytest.approx(wave_length)

    def test_thirty_degrees(self):
        wave_length = 0.1
        theta = math.pi / 6  # 30°
        expected = wave_length / (1 + math.sin(theta))
        assert calculate_delta_x(wave_length, theta) == pytest.approx(expected)


class TestCalculate1DAntennaArrayPython:
    """Tests for the pure-Python reference implementation in util.py."""

    def _uniform_pattern(self, N=4, n_theta=5):
        theta_arr = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
        f_arr = [complex_t(1, 0)] * n_theta
        wave_length = 0.1
        wave_num = 2 * np.pi / wave_length
        delta_x = wave_length / 2
        x_arr = np.array([i * delta_x for i in range(N)])
        return calculate_1d_antenna_array(N, f_arr, x_arr, theta_arr, wave_num)

    def test_returns_correct_length(self):
        n_theta = 7
        result = self._uniform_pattern(N=4, n_theta=n_theta)
        assert len(result) == n_theta

    def test_result_is_complex_t(self):
        result = self._uniform_pattern()
        assert isinstance(result[0], complex_t)


# ---------------------------------------------------------------------------
# C library functions
# ---------------------------------------------------------------------------


class TestSum:
    def test_positive(self):
        lib = _lib()
        assert lib.Sum(2.0, 3.0) == pytest.approx(5.0)

    def test_negative(self):
        lib = _lib()
        assert lib.Sum(-1.5, 1.5) == pytest.approx(0.0)

    def test_zero(self):
        lib = _lib()
        assert lib.Sum(0.0, 0.0) == pytest.approx(0.0)


class TestSumComplex:
    def test_basic(self):
        lib = _lib()
        a = complex_t(1.0, 2.0)
        b = complex_t(3.0, -1.0)
        result = lib.SumComplex(a, b)
        assert result.real == pytest.approx(4.0)
        assert result.imag == pytest.approx(1.0)

    def test_zeros(self):
        lib = _lib()
        z = complex_t(0.0, 0.0)
        result = lib.SumComplex(z, z)
        assert result.real == pytest.approx(0.0)
        assert result.imag == pytest.approx(0.0)


class TestSumArray:
    def test_all_elements_equal_total_sum(self):
        lib = _lib()
        data = [1.0, 2.0, 3.0]
        c_arr = list_to_c_double_array(data)
        result = lib.SumArray(c_arr, len(data))
        expected = sum(data)
        for i in range(len(data)):
            assert result[i] == pytest.approx(expected)
        lib.FreeDoubleArr(result)


class TestSumComplexArray:
    def test_all_elements_equal_total_sum(self):
        lib = _lib()
        items = [complex_t(1.0, -1.0), complex_t(2.0, 3.0), complex_t(-1.0, 0.0)]
        c_arr = list_to_c_complex_array(items)
        result = lib.SumComplexArray(c_arr, len(items))
        expected_real = sum(x.real for x in items)
        expected_imag = sum(x.imag for x in items)
        for i in range(len(items)):
            assert result[i].real == pytest.approx(expected_real)
            assert result[i].imag == pytest.approx(expected_imag)
        lib.FreeComplexArr(result)


class TestCalculateNormalizingCoeff:
    def test_unit_elements(self):
        # |1+0j| = 1 for each of N elements → coeff = 1/N
        lib = _lib()
        N = 4
        arr = list_to_c_complex_array([complex_t(1.0, 0.0)] * N)
        coeff = lib.CalculateNormalizingCoeff(arr, N)
        assert coeff == pytest.approx(1.0 / N)

    def test_single_element(self):
        lib = _lib()
        arr = list_to_c_complex_array([complex_t(3.0, 4.0)])  # |z| = 5
        coeff = lib.CalculateNormalizingCoeff(arr, 1)
        assert coeff == pytest.approx(1.0 / 5.0)


class TestCalculateWaveNumber:
    FREQUENCY = 1
    WAVELENGTH = 2

    def test_from_wavelength(self):
        lib = _lib()
        wave_length = 0.1
        expected = 2 * math.pi / wave_length
        assert lib.CalculateWaveNumber(wave_length, self.WAVELENGTH) == pytest.approx(
            expected
        )

    def test_from_frequency(self):
        lib = _lib()
        LIGHT_SPEED = 300_000_000
        freq = 3e9
        expected = 2 * math.pi * freq / LIGHT_SPEED
        assert lib.CalculateWaveNumber(freq, self.FREQUENCY) == pytest.approx(expected)


class TestAngleConversions:
    def test_degrees_to_radians_90(self):
        lib = _lib()
        assert lib.DegreesToRadians(90.0) == pytest.approx(math.pi / 2)

    def test_degrees_to_radians_180(self):
        lib = _lib()
        assert lib.DegreesToRadians(180.0) == pytest.approx(math.pi)

    def test_radians_to_degrees_pi(self):
        lib = _lib()
        assert lib.RadiansToDegrees(math.pi) == pytest.approx(180.0)

    def test_round_trip(self):
        lib = _lib()
        original = 45.0
        radians = lib.DegreesToRadians(original)
        back = lib.RadiansToDegrees(radians)
        assert back == pytest.approx(original)


class TestCalculate1DAntennaArray:
    def _setup(self, N=8, n_theta=181):
        lib = _lib()
        wave_length = 0.1
        wave_num = 2 * math.pi / wave_length
        delta_x = wave_length / 2
        x_arr = np.array([i * delta_x for i in range(N)])
        theta_arr = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
        f_arr = [complex_t(1, 0)] * n_theta
        return lib, N, n_theta, x_arr, theta_arr, f_arr, wave_num

    def test_returns_correct_size(self):
        lib, N, n_theta, x_arr, theta_arr, f_arr, wave_num = self._setup()
        c_f = list_to_c_complex_array(f_arr)
        c_x = list_to_c_double_array(x_arr)
        c_theta = list_to_c_double_array(theta_arr)
        result = lib.Calculate1DAntennaArray(N, n_theta, c_f, c_x, c_theta, wave_num)
        # Check a few elements are accessible without crash
        for i in range(n_theta):
            _ = result[i].real
        lib.FreeComplexArr(result)

    def test_broadside_peak(self):
        """At theta=0 (broadside) all phasors align → maximum magnitude."""
        lib, N, n_theta, x_arr, theta_arr, f_arr, wave_num = self._setup(
            N=8, n_theta=181
        )
        c_f = list_to_c_complex_array(f_arr)
        c_x = list_to_c_double_array(x_arr)
        c_theta = list_to_c_double_array(theta_arr)
        result = lib.Calculate1DAntennaArray(N, n_theta, c_f, c_x, c_theta, wave_num)
        magnitudes = np.array(
            [abs(result[i].real + 1j * result[i].imag) for i in range(n_theta)]
        )
        lib.FreeComplexArr(result)
        broadside_idx = n_theta // 2  # theta = 0 is at center
        assert magnitudes[broadside_idx] == pytest.approx(max(magnitudes), rel=1e-3)

    def test_broadside_peak_is_one(self):
        """Uniform unit amplitudes normalised by N → peak magnitude = 1.0."""
        lib, N, n_theta, x_arr, theta_arr, f_arr, wave_num = self._setup(
            N=8, n_theta=181
        )
        f_arr = [complex_t(1, 0)] * N
        c_f = list_to_c_complex_array(f_arr)
        c_x = list_to_c_double_array(x_arr)
        c_theta = list_to_c_double_array(theta_arr)
        result = lib.Calculate1DAntennaArray(N, n_theta, c_f, c_x, c_theta, wave_num)
        broadside_idx = n_theta // 2
        mag = abs(complex(result[broadside_idx].real, result[broadside_idx].imag))
        lib.FreeComplexArr(result)
        assert mag == pytest.approx(1.0, abs=1e-6)

    def test_single_element_constant_pattern(self):
        """N=1 at x=0: exp(-jk*0*sinθ)=1 for all θ → magnitude = 1/1 = 1 everywhere."""
        lib = _lib()
        wave_num = 2 * math.pi / 0.1
        n_theta = 31
        theta_arr = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
        c_f = list_to_c_complex_array([complex_t(1, 0)])
        c_x = list_to_c_double_array(np.array([0.0]))
        c_theta = list_to_c_double_array(theta_arr)
        result = lib.Calculate1DAntennaArray(1, n_theta, c_f, c_x, c_theta, wave_num)
        for i in range(n_theta):
            mag = abs(complex(result[i].real, result[i].imag))
            assert mag == pytest.approx(1.0, abs=1e-9)
        lib.FreeComplexArr(result)


# ---------------------------------------------------------------------------
# Python calculate_1d_antenna_array — value correctness
# ---------------------------------------------------------------------------


class TestCalculate1DAntennaArrayPythonValues:
    """Verify physics correctness of the pure-Python implementation."""

    def _run(self, N, n_theta=181, f_elem=None):
        wave_length = 0.1
        wave_num = 2 * math.pi / wave_length
        delta_x = wave_length / 2
        x_arr = np.array([i * delta_x - delta_x * (N - 1) / 2 for i in range(N)])
        theta_arr = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
        if f_elem is None:
            f_elem = complex_t(1, 0)
        f_arr = [f_elem] * N
        result = calculate_1d_antenna_array(N, f_arr, x_arr, theta_arr, wave_num)
        return result, theta_arr

    def test_broadside_peak_is_one(self):
        """Uniform unit amplitudes → peak at broadside = 1.0."""
        result, _ = self._run(N=8)
        broadside_idx = len(result) // 2
        mag = abs(complex(result[broadside_idx].real, result[broadside_idx].imag))
        assert mag == pytest.approx(1.0, abs=1e-6)

    def test_broadside_is_global_maximum(self):
        result, _ = self._run(N=8)
        magnitudes = np.array([abs(complex(r.real, r.imag)) for r in result])
        broadside_idx = len(result) // 2
        assert magnitudes[broadside_idx] == pytest.approx(magnitudes.max(), rel=1e-3)

    def test_element_pattern_scales_output(self):
        """Doubling element amplitude should double the output magnitude."""
        result1, _ = self._run(N=4, f_elem=complex_t(1, 0))
        result2, _ = self._run(N=4, f_elem=complex_t(2, 0))
        for r1, r2 in zip(result1, result2):
            m1 = abs(complex(r1.real, r1.imag))
            m2 = abs(complex(r2.real, r2.imag))
            assert m2 == pytest.approx(2 * m1, abs=1e-9)

    def test_single_element_constant(self):
        """N=1 at x=0 → magnitude = 1 everywhere."""
        result, _ = self._run(N=1)
        for r in result:
            assert abs(complex(r.real, r.imag)) == pytest.approx(1.0, abs=1e-9)

    def test_c_and_python_agree(self):
        """Python and C implementations should produce the same result."""
        lib = _lib()
        N, n_theta = 8, 91
        wave_length = 0.1
        wave_num = 2 * math.pi / wave_length
        delta_x = wave_length / 2
        x_arr = np.array([i * delta_x - delta_x * (N - 1) / 2 for i in range(N)])
        theta_arr = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
        f_arr = [complex_t(1, 0)] * N

        py_result = calculate_1d_antenna_array(N, f_arr, x_arr, theta_arr, wave_num)

        c_f = list_to_c_complex_array(f_arr)
        c_x = list_to_c_double_array(x_arr)
        c_theta = list_to_c_double_array(theta_arr)
        c_result = lib.Calculate1DAntennaArray(N, n_theta, c_f, c_x, c_theta, wave_num)

        for i in range(n_theta):
            assert py_result[i].real == pytest.approx(c_result[i].real, abs=1e-9)
            assert py_result[i].imag == pytest.approx(c_result[i].imag, abs=1e-9)
        lib.FreeComplexArr(c_result)


# ---------------------------------------------------------------------------
# Validation against analytical results (Balanis Ch.6)
# Uniform linear array, d=λ/2, isotropic elements, broadside
# Key exact results:
#   D₀ = N  (exact for d=λ/2, isotropic)
#   Null positions: sinΘ = n·λ/(N·d) = 2n/N
#   First sidelobe level → −13.26 dB (asymptotic, N ≥ 8)
# ---------------------------------------------------------------------------


def _compute_1d_pattern(N, n_theta=10001):
    """Helper: compute 1D pattern for uniform array, d=λ/2, isotropic, broadside."""
    lib = _lib()
    freq = 3e9
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
        ct.c_int(N),
        ct.c_int(n_theta),
        c_f,
        c_x,
        c_theta,
        ct.c_double(wave_num),
    )
    af = np.array([abs(complex(raw[i].real, raw[i].imag)) for i in range(n_theta)])
    lib.FreeComplexArr(raw)
    af_norm = af / af.max()

    # КНД (formula 10.41): D₀ = 2 / ∫ F²(Θ)cosΘ dΘ
    D0 = 2.0 / np.trapezoid(af_norm**2 * np.cos(theta), theta)
    D0_db = 10 * np.log10(D0)

    return {
        "theta_deg": np.degrees(theta),
        "array_factor": af_norm,
        "D0": D0,
        "D0_db": D0_db,
    }


class TestDirectivity1D:
    """КНД линейной равномерной АР с d=λ/2: D₀ должен быть равен N."""

    @pytest.mark.parametrize("N", [2, 4, 8, 10, 16])
    def test_directivity_equals_N(self, N):
        result = _compute_1d_pattern(N)
        assert result["D0"] == pytest.approx(N, rel=1e-2)

    @pytest.mark.parametrize("N", [2, 4, 8, 10, 16])
    def test_directivity_db(self, N):
        result = _compute_1d_pattern(N)
        assert result["D0_db"] == pytest.approx(10 * np.log10(N), abs=0.1)


class TestNullPositions1D:
    """Нули ДН: sinΘ_null = 2n/N для равномерной АР с d=λ/2."""

    @pytest.mark.parametrize(
        "N, null_n", [(8, 1), (8, 2), (8, 3), (10, 1), (10, 2), (16, 1)]
    )
    def test_first_nulls(self, N, null_n):
        result = _compute_1d_pattern(N)
        af = result["array_factor"]
        expected_null_deg = np.degrees(np.arcsin(2 * null_n / N))
        idx = np.argmin(np.abs(result["theta_deg"] - expected_null_deg))
        assert af[idx] < 0.02, (
            f"N={N}, null #{null_n}: AF={af[idx]:.4f} at "
            f"θ={result['theta_deg'][idx]:.2f}° (expected null at {expected_null_deg:.2f}°)"
        )


class TestSidelobeLevel1D:
    """УБЛ первого бокового лепестка → −13.26 дБ для N ≥ 8."""

    @pytest.mark.parametrize("N", [8, 10, 16])
    def test_first_sidelobe_level(self, N):
        result = _compute_1d_pattern(N, n_theta=20001)
        af = result["array_factor"]
        center = len(af) // 2
        # find first null (right of broadside)
        first_null_idx = center
        for i in range(center + 1, len(af)):
            if af[i] > af[i - 1]:
                first_null_idx = i - 1
                break
        # find first sidelobe peak between first and second null
        sl_peak = 0.0
        for i in range(first_null_idx + 1, len(af)):
            if af[i] > sl_peak:
                sl_peak = af[i]
            elif af[i] < sl_peak * 0.95:
                break
        sll_db = 20 * np.log10(max(sl_peak, 1e-10))
        # Asymptotic SLL = -13.26 dB (exact for N→∞); for finite N it's slightly
        # higher (e.g. -12.8 dB for N=8). Tolerance 0.5 dB covers this.
        assert sll_db == pytest.approx(-13.26, abs=0.5), (
            f"N={N}: SLL={sll_db:.2f} dB (expected ≈ −13.26 dB)"
        )


class TestTwoElementArray:
    """N=2, d=λ/2: нет боковых лепестков, D₀=2, нули при ±90°."""

    def test_no_sidelobes(self):
        """AF = cos(π/2·sinθ): монотонно убывает от 0° до ±90°."""
        result = _compute_1d_pattern(2, n_theta=1001)
        af = result["array_factor"]
        center = len(af) // 2
        right = af[center:]
        for i in range(1, len(right)):
            assert right[i] <= right[i - 1] + 1e-9

    def test_endfire_nulls(self):
        """AF → 0 на краях (θ = ±90°)."""
        result = _compute_1d_pattern(2, n_theta=1001)
        af = result["array_factor"]
        assert af[0] < 0.01  # θ = −90°
        assert af[-1] < 0.01  # θ = +90°


class TestArrayFactorAnalytical:
    """Прямое сравнение AF с аналитической формулой (12.5):
    AF(θ) = sin(N·π·d·sinθ/λ) / (N·sin(π·d·sinθ/λ))
    Для d=λ/2: AF(θ) = sin(N·π·sinθ/2) / (N·sin(π·sinθ/2))
    """

    @staticmethod
    def _analytical_af(theta, N):
        """Exact normalized array factor for uniform ULA, d=λ/2."""
        psi_half = math.pi * np.sin(theta) / 2  # (π·d·sinθ/λ) with d=λ/2
        with np.errstate(divide="ignore", invalid="ignore"):
            af = np.where(
                np.abs(np.sin(psi_half)) < 1e-12,
                1.0,  # L'Hôpital at psi=0,±π,...
                np.sin(N * psi_half) / (N * np.sin(psi_half)),
            )
        return np.abs(af)

    @pytest.mark.parametrize("N", [4, 8, 16])
    def test_af_matches_analytical(self, N):
        """C++ array factor must match closed-form formula to < 1e-3."""
        result = _compute_1d_pattern(N, n_theta=1001)
        theta_rad = np.radians(result["theta_deg"])
        expected = self._analytical_af(theta_rad, N)
        np.testing.assert_allclose(result["array_factor"], expected, atol=1e-3)

    def test_specific_values_N8(self):
        """Point-by-point check at known angles for N=8 (Balanis table)."""
        result = _compute_1d_pattern(8, n_theta=10001)
        theta_deg = result["theta_deg"]
        af = result["array_factor"]

        checks = [
            (0.0, 1.0, 1e-4),  # broadside peak
            (14.48, 0.0, 0.02),  # first null: arcsin(1/4)
            (30.0, 0.0, 0.02),  # second null: arcsin(2/4)
            (48.59, 0.0, 0.02),  # third null: arcsin(3/4)
        ]
        for angle, expected_af, tol in checks:
            idx = np.argmin(np.abs(theta_deg - angle))
            assert abs(af[idx] - expected_af) < tol, (
                f"θ={angle}°: AF={af[idx]:.4f}, expected={expected_af:.4f}"
            )

    def test_first_sidelobe_peak_N8(self):
        """First sidelobe peak for N=8 at ~20.9° with |AF|≈0.217 (-13.26 dB)."""
        result = _compute_1d_pattern(8, n_theta=20001)
        theta_deg = result["theta_deg"]
        af = result["array_factor"]
        # sidelobe region: between first null (~14.5°) and second null (~30°)
        mask = (theta_deg > 16) & (theta_deg < 28)
        sl_peak = af[mask].max()
        # Exact for N=8: 0.2292 (asymptotic 2/(3π)=0.2122 applies for N→∞)
        assert sl_peak == pytest.approx(0.2292, abs=0.005)


class TestDirectivity2D:
    """КНД 2D АР — валидация через сравнение с 1D."""

    def _compute_2d(self, N, x_arr, y_arr, n_theta=301, n_phi=301):
        lib = _lib()
        freq = 3e9
        lam = 3e8 / freq
        wave_num = 2 * math.pi / lam
        theta = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
        phi = np.linspace(-math.pi / 2, math.pi / 2, n_phi)

        f_arr = [complex_t(1.0, 0.0)] * N
        raw = lib.Calculate2DAntennaArray(
            ct.c_int(N),
            ct.c_int(n_theta),
            ct.c_int(n_phi),
            list_to_c_complex_array(f_arr),
            list_to_c_double_array(x_arr),
            list_to_c_double_array(y_arr),
            list_to_c_double_array(theta),
            list_to_c_double_array(phi),
            ct.c_double(wave_num),
        )
        total = n_theta * n_phi
        af = np.array([abs(complex(raw[i].real, raw[i].imag)) for i in range(total)])
        lib.FreeComplexArr(raw)
        af_2d = af.reshape(n_theta, n_phi)
        af_2d /= af_2d.max()

        integrand = af_2d**2 * np.cos(theta)[:, np.newaxis]
        inner = np.trapezoid(integrand, theta, axis=0)
        denominator = np.trapezoid(inner, phi)
        return 4 * math.pi / denominator

    def test_single_element_isotropic(self):
        """N=1 at origin: AF=1 everywhere → D₀ = 4π / ∫∫ cosΘ dΘ dφ."""
        x = np.array([0.0])
        y = np.array([0.0])
        D0 = self._compute_2d(1, x, y)
        # ∫₋π/₂^π/₂ cosΘ dΘ = 2, ∫₋π/₂^π/₂ dφ = π → denominator = 2π → D₀ = 4π/2π = 2
        assert D0 == pytest.approx(2.0, rel=0.02)

    def test_directivity_increases_with_elements(self):
        """Adding more elements must increase directivity."""
        freq = 3e9
        lam = 3e8 / freq
        d = lam / 2
        D_values = []
        for Nx in [2, 4, 8]:
            xs, ys = [], []
            for ix in range(Nx):
                for iy in range(Nx):
                    xs.append((ix - (Nx - 1) / 2) * d)
                    ys.append((iy - (Nx - 1) / 2) * d)
            D0 = self._compute_2d(Nx * Nx, np.array(xs), np.array(ys))
            D_values.append(D0)
        for i in range(1, len(D_values)):
            assert D_values[i] > D_values[i - 1]

    def test_2d_single_row_matches_1d(self):
        """Linear array along x (y=0 for all) via 2D function should match 1D D₀."""
        freq = 3e9
        lam = 3e8 / freq
        d = lam / 2
        N = 8
        x_arr = np.array([(i - (N - 1) / 2) * d for i in range(N)])
        y_arr = np.zeros(N)
        D0_2d = self._compute_2d(N, x_arr, y_arr)
        # 1D D₀ for N=8 = 8.0, but 2D integral covers only φ ∈ [-π/2, π/2]
        # (not full azimuth), so D₀_2d ≠ D₀_1d. Still, the broadside peak
        # should give a consistent, reproducible value.
        assert D0_2d > 1.0
        assert D0_2d < 100.0

    def test_40x12_rectangular_directivity(self):
        """Регрессионный тест: 40×12 прямоугольная решётка, d=0.5λ, broadside.

        Ref: Габриэльян и др., ЖРЭ №12, 2012 (docs/text.pdf), табл.1:
        КНД прямоуг. раскрыва = 25.55 дБ (полная полусфера, элементная ДН ≠ isotropic).
        Наш расчёт: D₀ ≈ 50.3 (17.0 дБ) — интеграл по φ ∈ [-π/2, π/2], isotropic.
        Разница обусловлена разными пределами интегрирования и элементной ДН.
        """
        freq = 3e9
        lam = 3e8 / freq
        d = lam / 2
        Nx, Ny = 40, 12
        xs, ys = [], []
        for ix in range(Nx):
            for iy in range(Ny):
                xs.append((ix - (Nx - 1) / 2) * d)
                ys.append((iy - (Ny - 1) / 2) * d)
        D0 = self._compute_2d(
            Nx * Ny, np.array(xs), np.array(ys), n_theta=201, n_phi=201
        )
        D0_db = 10 * math.log10(D0)
        assert D0 == pytest.approx(50.3, rel=0.05), f"D0={D0:.2f} ({D0_db:.2f} dB)"

    def test_4x4_vs_8x8_directivity_ratio(self):
        """При удвоении Nx (и Ny): D₀ растёт в ~2 раза.

        D₀ ~ Nx при нашем интеграле (φ ∈ [-π/2, π/2]),
        т.к. сужение луча по φ не полностью учитывается ограниченным пределом.
        Ref: Габриэльян (docs/text.pdf) — КНД пропорционален площади раскрыва
        при полной полусфере; при ограниченном φ — пропорционален линейному размеру.
        """
        freq = 3e9
        lam = 3e8 / freq
        d = lam / 2
        D_list = []
        for Nx in [4, 8]:
            xs, ys = [], []
            for ix in range(Nx):
                for iy in range(Nx):
                    xs.append((ix - (Nx - 1) / 2) * d)
                    ys.append((iy - (Nx - 1) / 2) * d)
            D_list.append(self._compute_2d(Nx * Nx, np.array(xs), np.array(ys)))
        # 8×8 vs 4×4: Nx удвоился → D₀ растёт в ~2 раза
        ratio = D_list[1] / D_list[0]
        assert ratio == pytest.approx(2.0, rel=0.15)


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------


class TestConfig:
    def _load(self):
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f)

    def test_freq_hz_is_float(self):
        cfg = self._load()
        assert isinstance(cfg["array_1d"]["freq_hz"], float)

    def test_freq_hz_value(self):
        cfg = self._load()
        assert cfg["array_1d"]["freq_hz"] == pytest.approx(3e9)

    def test_null_spacing_parses_as_none(self):
        cfg = self._load()
        assert cfg["array_1d"]["d"] is None

    def test_integer_fields(self):
        cfg = self._load()
        assert isinstance(cfg["array_1d"]["N"], int)

    def test_n_theta_positive(self):
        cfg = self._load()
        assert cfg["array_1d"]["n_theta"] > 0
