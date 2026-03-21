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

from python.complex import complex_t
from python.util import (
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

_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
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
        assert lib.CalculateWaveNumber(wave_length, self.WAVELENGTH) == pytest.approx(expected)

    def test_from_frequency(self):
        lib = _lib()
        LIGHT_SPEED = 300_000_000
        freq = 3e9
        expected = 2 * math.pi * freq / LIGHT_SPEED
        assert lib.CalculateWaveNumber(freq, self.FREQUENCY) == pytest.approx(expected)


class TestCalculateAntennaArraySize:
    def test_basic(self):
        lib = _lib()
        assert lib.CalculateAntennaArraySize(4, 4) == 16

    def test_linear(self):
        lib = _lib()
        assert lib.CalculateAntennaArraySize(8, 1) == 8


class TestCalculateDelta:
    def test_delta_x_broadside(self):
        lib = _lib()
        wave_length = 0.1
        result = lib.CalculateDeltaX(wave_length, 0.0)
        assert result == pytest.approx(wave_length)

    def test_delta_y_broadside(self):
        lib = _lib()
        wave_length = 0.1
        result = lib.CalculateDeltaY(wave_length, 0.0)
        assert result == pytest.approx(wave_length)

    def test_delta_x_thirty_degrees(self):
        lib = _lib()
        wave_length = 0.1
        theta = math.pi / 6
        expected = wave_length / (1 + math.sin(theta))
        assert lib.CalculateDeltaX(wave_length, theta) == pytest.approx(expected)


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
        lib, N, n_theta, x_arr, theta_arr, f_arr, wave_num = self._setup(N=8, n_theta=181)
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
