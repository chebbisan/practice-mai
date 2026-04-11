"""
Tests for the beam analysis module (app/analysis.py).

Verifies extracted beam parameters against analytical values for
uniform linear arrays with d=λ/2, isotropic elements.

Run with:
    python -m pytest tests/test_analysis.py -v
"""

import math
from pathlib import Path

import numpy as np
import pytest

from app.analysis import analyze_cut, analyze_pattern_1d, analyze_pattern_2d, analyze_csv


# ---------------------------------------------------------------------------
# Helpers: generate synthetic 1D pattern
# ---------------------------------------------------------------------------


def _make_1d_pattern(N, n_theta=20001):
    """Uniform array, d=λ/2, isotropic, broadside. Returns PatternResult-like dict."""
    freq = 3e9
    lam = 3e8 / freq
    d = lam / 2
    k = 2 * math.pi / lam
    L = d * (N - 1)
    x_arr = np.array([i * d - L / 2 for i in range(N)])

    theta = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
    # NumPy vectorized AF
    phase = -k * np.outer(x_arr, np.sin(theta))
    af = np.abs(np.sum(np.exp(1j * phase), axis=0)) / N

    af /= af.max()
    pattern_db = 20 * np.log10(np.maximum(af, 1e-10))

    D0 = 2.0 / np.trapezoid(af**2 * np.cos(theta), theta)

    return {
        "theta_deg": np.degrees(theta),
        "pattern_db": pattern_db,
        "array_factor": af,
        "N": N,
        "D0": D0,
        "D0_db": 10 * np.log10(D0),
    }


def _make_2d_pattern(Nx, Ny, n_theta=201, n_phi=201):
    """Rectangular array Nx×Ny, d=λ/2, isotropic. Returns compute_pattern_2d-like dict."""
    freq = 3e9
    lam = 3e8 / freq
    d = lam / 2
    k = 2 * math.pi / lam

    xs, ys = [], []
    for ix in range(Nx):
        for iy in range(Ny):
            xs.append((ix - (Nx - 1) / 2) * d)
            ys.append((iy - (Ny - 1) / 2) * d)
    x_arr = np.array(xs)
    y_arr = np.array(ys)
    N = len(x_arr)

    theta = np.linspace(-math.pi / 2, math.pi / 2, n_theta)
    phi = np.linspace(-math.pi / 2, math.pi / 2, n_phi)

    st_cp = np.outer(np.sin(theta), np.cos(phi))
    st_sp = np.outer(np.sin(theta), np.sin(phi))
    af_complex = np.zeros((n_theta, n_phi), dtype=complex)
    for n in range(N):
        af_complex += np.exp(-1j * k * (x_arr[n] * st_cp + y_arr[n] * st_sp))
    af_2d = np.abs(af_complex) / N
    af_2d /= af_2d.max()

    pattern_db = 20 * np.log10(np.maximum(af_2d, 1e-10))

    integrand = af_2d**2 * np.cos(theta)[:, np.newaxis]
    inner = np.trapezoid(integrand, theta, axis=0)
    D0 = 4 * math.pi / np.trapezoid(inner, phi)

    return {
        "theta": theta,
        "phi": phi,
        "pattern_2d": af_2d,
        "pattern_db": pattern_db,
        "N": N,
        "D0": D0,
        "D0_db": 10 * np.log10(D0),
    }


# ---------------------------------------------------------------------------
# Tests: analyze_cut on 1D patterns
# ---------------------------------------------------------------------------


class TestAnalyzeCut1D:
    """Верификация параметров главного лепестка для равномерной решётки."""

    def test_peak_at_broadside(self):
        result = _make_1d_pattern(16)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        assert a["peak_deg"] == pytest.approx(0.0, abs=0.1)

    @pytest.mark.parametrize("N", [8, 16, 32])
    def test_beamwidth_3db(self, N):
        """2Θ₀.₇ ≈ 51°·λ/(N·d).  Для d=λ/2: 2Θ₀.₇ ≈ 102°/N."""
        result = _make_1d_pattern(N)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        expected_bw = 102.0 / N
        # Формула приближённая; допуск 15%
        assert a["beamwidth_3db"] == pytest.approx(expected_bw, rel=0.15), (
            f"N={N}: BW3={a['beamwidth_3db']:.2f}° (expected ≈ {expected_bw:.2f}°)"
        )

    def test_beamwidth_10db_wider_than_3db(self):
        result = _make_1d_pattern(16)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        assert a["beamwidth_10db"] > a["beamwidth_3db"]

    @pytest.mark.parametrize("N", [8, 16])
    def test_first_null_position(self, N):
        """Первый нуль при sin(Θ) = 2/N → Θ = arcsin(2/N) для d=λ/2."""
        result = _make_1d_pattern(N)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        expected_null = math.degrees(math.asin(2.0 / N))
        assert a["first_null_right_deg"] == pytest.approx(expected_null, abs=0.3)
        # Симметрия
        assert a["first_null_left_deg"] == pytest.approx(-expected_null, abs=0.3)

    @pytest.mark.parametrize("N", [8, 10, 16])
    def test_sidelobe_level(self, N):
        """УБЛ → −13.26 дБ для N → ∞. Допуск 0.5 дБ для конечных N."""
        result = _make_1d_pattern(N)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        assert a["first_sll_db"] is not None
        assert a["first_sll_db"] == pytest.approx(-13.26, abs=0.5)

    @pytest.mark.parametrize("N", [8, 16])
    def test_sector_deg(self, N):
        """Сектор по нулям = 2·arcsin(2/N) для d=λ/2."""
        result = _make_1d_pattern(N)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        expected_sector = 2 * math.degrees(math.asin(2.0 / N))
        assert a["sector_deg"] == pytest.approx(expected_sector, abs=0.6)

    def test_sector_wider_than_beamwidth(self):
        result = _make_1d_pattern(16)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        assert a["sector_deg"] > a["beamwidth_3db"]

    def test_two_element_no_sidelobe(self):
        """N=2: AF = cos(π/2·sinθ) — нет боковых лепестков."""
        result = _make_1d_pattern(2, n_theta=10001)
        a = analyze_cut(result["theta_deg"], result["pattern_db"])
        # Боковых лепестков нет — функция должна вернуть None
        assert a["first_sll_db"] is None


# ---------------------------------------------------------------------------
# Tests: analyze_pattern_1d
# ---------------------------------------------------------------------------


class TestAnalyzePattern1D:
    def test_includes_directivity(self):
        result = _make_1d_pattern(16)
        a = analyze_pattern_1d(result)
        assert a["D0"] == pytest.approx(16.0, rel=0.01)
        assert a["D0_db"] == pytest.approx(10 * math.log10(16), rel=0.01)

    def test_returns_all_keys(self):
        result = _make_1d_pattern(8)
        a = analyze_pattern_1d(result)
        for key in ("peak_deg", "beamwidth_3db", "beamwidth_10db",
                     "first_null_left_deg", "first_null_right_deg",
                     "first_sll_db", "D0", "D0_db"):
            assert key in a


# ---------------------------------------------------------------------------
# Tests: analyze_pattern_2d
# ---------------------------------------------------------------------------


class TestAnalyzePattern2D:
    def test_rectangular_array_xz_narrower(self):
        """40×12: XZ срез (40 эл.) уже, чем YZ (12 эл.)."""
        result = _make_2d_pattern(40, 12, n_theta=401, n_phi=401)
        a = analyze_pattern_2d(result)
        bw_xz = a["cut_xz"]["beamwidth_3db"]
        bw_yz = a["cut_yz"]["beamwidth_3db"]
        assert bw_xz < bw_yz

    def test_square_array_symmetric(self):
        """8×8: XZ и YZ срезы должны давать одинаковую ширину луча."""
        result = _make_2d_pattern(8, 8)
        a = analyze_pattern_2d(result)
        bw_xz = a["cut_xz"]["beamwidth_3db"]
        bw_yz = a["cut_yz"]["beamwidth_3db"]
        assert bw_xz == pytest.approx(bw_yz, rel=0.05)

    def test_includes_directivity(self):
        result = _make_2d_pattern(4, 4)
        a = analyze_pattern_2d(result)
        assert a["D0"] is not None
        assert a["D0"] > 1.0

    def test_beam_solid_angle_3db(self):
        """Пространственный сектор (−3 дБ) для квадратной решётки."""
        result = _make_2d_pattern(8, 8)
        a = analyze_pattern_2d(result)
        omega = a["beam_solid_angle_3db_sr"]
        assert omega is not None
        assert omega > 0

    def test_beam_solid_angle_null(self):
        """Пространственный сектор (по нулям) шире, чем по −3 дБ."""
        result = _make_2d_pattern(8, 8)
        a = analyze_pattern_2d(result)
        assert a["beam_solid_angle_null_sr"] > a["beam_solid_angle_3db_sr"]

    def test_rectangular_solid_angle(self):
        """40×12: сектор по −3 дБ для прямоугольной решётки — не квадратный."""
        result = _make_2d_pattern(40, 12, n_theta=401, n_phi=401)
        a = analyze_pattern_2d(result)
        bw_xz = a["cut_xz"]["beamwidth_3db"]
        bw_yz = a["cut_yz"]["beamwidth_3db"]
        # Ω ≈ bw_xz_rad * bw_yz_rad
        expected = math.radians(bw_xz) * math.radians(bw_yz)
        assert a["beam_solid_angle_3db_sr"] == pytest.approx(expected, rel=0.01)


# ---------------------------------------------------------------------------
# Tests: analyze_csv
# ---------------------------------------------------------------------------


class TestAnalyzeCSV:
    def test_1d_csv(self, tmp_path):
        """Анализ 1D CSV даёт те же результаты, что и анализ в памяти."""
        result = _make_1d_pattern(16)
        a_mem = analyze_cut(result["theta_deg"], result["pattern_db"])

        # Записываем CSV в формате экспорта
        csv_path = tmp_path / "test_1d.csv"
        with open(csv_path, "w") as f:
            f.write("# Test export\n")
            f.write("Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]\n")
            for i, t in enumerate(result["theta_deg"]):
                f.write(f"{t:.4f}; 0.0000; {result['pattern_db'][i]:.4f}\n")

        a_csv = analyze_csv(csv_path)
        assert a_csv["beamwidth_3db"] == pytest.approx(a_mem["beamwidth_3db"], rel=0.01)
        assert a_csv["first_sll_db"] == pytest.approx(a_mem["first_sll_db"], abs=0.1)

    def test_2d_csv(self, tmp_path):
        """Анализ 2D CSV даёт результат с двумя срезами."""
        result = _make_2d_pattern(8, 8, n_theta=101, n_phi=101)
        theta_deg = np.degrees(result["theta"])
        phi_deg = np.degrees(result["phi"])

        csv_path = tmp_path / "test_2d.csv"
        with open(csv_path, "w") as f:
            f.write("# Test 2D export\n")
            f.write("Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]\n")
            for j, p in enumerate(phi_deg):
                for i, t in enumerate(theta_deg):
                    f.write(f"{t:.4f}; {p:.4f}; {result['pattern_db'][i, j]:.4f}\n")

        a_csv = analyze_csv(csv_path)
        assert "cut_xz" in a_csv
        assert "cut_yz" in a_csv
        assert a_csv["cut_xz"]["beamwidth_3db"] is not None
