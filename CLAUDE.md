# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Python/C++ project for calculating and analyzing radiation patterns of phased antenna arrays (1D linear, 2D planar, 3D spatial). NumPy handles all primary computations (vectorized, BLAS/SIMD). C++ shared library retained for benchmarking only.

## Build

Requires CMake 3.28+ and a C++ compiler (only for benchmarks).

```bash
mkdir build && cd build
cmake ..
make
```

This produces:
- `build/Practice` — standalone executable
- `build/libAntennaArray.so` / `libAntennaArray.dylib` (macOS) — shared library for benchmarks

Alternatively, use `run.sh` which builds automatically before running any target.

## Running

```bash
# Quick launcher (builds if needed)
./run.sh {1d|2d|3d|app|bench|test}

# Direct Python scripts
cd app && python calc_1d.py    # 1D radiation pattern
cd app && python calc_2d.py    # 2D radiation pattern from CSV
cd app && python calc_3d.py    # 3D spatial array from CSV
cd app && python gui.py        # PyQt6 GUI
```

## Architecture

### Python Layer (`app/`)
- `common.py` — shared utilities: `SPEED_OF_LIGHT`, `setup_logging()`, `load_config()`, `load_array_from_csv()`, `element_pattern()`, `export_pattern_2d_csv()`
- `analysis.py` — beam analysis: `analyze_cut()`, `analyze_pattern_1d()`, `analyze_pattern_2d()`, `analyze_csv()`, `format_analysis()`
- `calc_1d.py` — 1D radiation pattern: `compute_pattern()` (NumPy vectorized), `export_pattern_csv()`, plotting
- `calc_2d.py` — 2D/3D radiation pattern: `compute_pattern_2d(z_arr=None)` (NumPy vectorized), plotting
- `calc_3d.py` — 3D wrapper: `compute_pattern_3d()` delegates to `compute_pattern_2d(z_arr=...)`
- `plot_2d.py` — polar heatmap visualization
- `gui.py` — PyQt6 GUI with tabs: 1D, 2D heatmap, 3D surface, 3D sphere
- `util.py` — loads C shared library, declares C function signatures, array conversion helpers
- `complex.py` — mirrors the C++ `complex_t` struct as a `ctypes.Structure`
- `config.yaml` — simulation parameters; read by `load_config(section)`
- `input/` — input CSV files (element positions + amplitudes)
- `output/` — exported radiation patterns in CST-compatible CSV format

### C++ Core (`lib/`) — used for benchmarks
- `antenna_array.hpp/cpp` — core physics:
  - `Calculate1DAntennaArray()`, `Calculate2DAntennaArray()`, `Calculate3DAntennaArray()` — array factor calculations
  - `FullPipeline1D()`, `FullPipeline2D()`, `FullPipeline3D()` — complete pipeline in single call (for pure C++ benchmarks)
  - `ComplexArrayMagnitude()`, `NormalizeArray()`, `CalculateDirectivity1D()`, `CalculateDirectivity2D()` — utilities
- `complex.hpp/cpp` — custom `complex_t` struct `{double real, imag}`
- `sum.hpp/cpp` — array summation utilities

## Configuration

`app/config.yaml` controls all simulation parameters:
- `array_1d`: N, freq_hz, d (null → λ/(1+sin(steer_deg))), steer_deg, n_theta, element_pattern, csv_file
- `array_2d`: freq_hz, n_theta, n_phi, element_pattern, csv_file (3-column CSV: x_m, y_m, amplitude_db)
- `array_3d`: freq_hz, n_theta, n_phi, element_pattern, csv_file (4-column CSV: x_m, y_m, z_m, amplitude_db)

### CSV Input Format
- 1D: `x_m, amplitude_db` (2 columns)
- 2D: `x_m, y_m, amplitude_db` (3 columns)
- 3D: `x_m, y_m, z_m, amplitude_db` (4 columns)
- Auto-detected by `load_array_from_csv()` in `common.py` based on column count

### CSV Output Format (CST-compatible)
Columns: `Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]; Abs(F); Phase(F)[deg.]`
Semicolon-separated, with header comments for frequency, N, directivity.

## Tests

```bash
python -m pytest tests/test_antenna_array.py -v   # 76 unit tests + cross-dimensional
python -m pytest tests/test_analysis.py -v         # 32 beam analysis tests
python -m pytest tests/test_benchmark.py -v        # individual operation benchmarks
python -m pytest tests/test_benchmark_1d.py -v     # full 1D pipeline: Python vs NumPy vs C++ vs Pure C++
python -m pytest tests/test_benchmark_2d.py -v     # full 2D pipeline: Python vs NumPy vs C++ vs Pure C++
python -m pytest tests/test_benchmark_3d.py -v     # full 3D pipeline: Python vs NumPy vs C++ vs Pure C++
```

### Linting
```bash
ruff check --ignore F403,F405 .   # F403/F405 only in legacy notebooks
ruff format .
clang-format -i lib/*.cpp lib/*.hpp src/*.cpp
```

## Physics Reference (Chapter 12 — Antenna Arrays)

Source: `docs/stuff.pdf`, pages 152–166.

### 1D Linear Array — Array Factor

Normalized array factor (formula 12.5):

```
fₙ(Θ) = sin(π·N·d·sinΘ / λ) / (N · sin(π·d·sinΘ / λ))
```

Key properties:
- Main lobe at Θ = 0° (broadside), first sidelobe: −13.26 dB (asymptotic, N ≥ 12)
- **Single main lobe condition**: `d < λ`, practically `d ≤ λ/2`
- Half-power beamwidth: `2Θ₀.₇ ≈ 51°·λ/(N·d)`
- Null directions: `sinΘ = p·λ/(N·d)`, p = ±1, ±2, ...
- For d=λ/2, isotropic elements: **D₀ = N exactly** (verified in tests to machine precision)

### 2D Planar Array — Arbitrary Elements

```
F(Θ,φ) = (1/N) Σ fₙ · exp(-jk(xₙ sinΘ cosφ + yₙ sinΘ sinφ))
```

Matches formula (3) from Gabrielyan et al., JRE 12/2012 (`docs/text.pdf`).

### 3D Spatial Array — Arbitrary Elements

```
F(Θ,φ) = (1/N) Σ fₙ · exp(-jk(xₙ sinΘ cosφ + yₙ sinΘ sinφ + zₙ cosΘ))
```

Based on formula 12.9 from `docs/stuff.pdf`, section 12.4.

### Directivity — КНД (Section 10.3, formulas 10.39–10.41)

Source: `docs/stuff.pdf`, pages 130–131.

1D (axial symmetry, formula 10.41): `D₀ = 2 / ∫ F²(Θ) cosΘ dΘ`
2D/3D (formula 10.40): `D₀ = 4π / ∫∫ F²(Θ,φ) cosΘ dΘ dφ`

Computed via `np.trapezoid`.

### Beam Steering (Section 12.5)

Maximum element spacing for single steered beam (formula 12.15):
```
d < λ / (1 + sin(Θₘ))
```
Used when `d = null` in config.

### Mapping to Code

| Theory | Python / C++ |
|--------|-------------|
| `k = 2π/λ` | `2 * np.pi / wave_length` (Python), `CalculateWaveNumber(freq_hz)` (C++) |
| `d < λ/(1+sinΘₘ)` | `calculate_delta_x(wave_length, steer)` in `util.py` |
| 1D array factor | `compute_pattern()` in `calc_1d.py` |
| 2D array factor | `compute_pattern_2d()` in `calc_2d.py` |
| 3D array factor | `compute_pattern_3d()` in `calc_3d.py` → `compute_pattern_2d(z_arr=...)` |
| `D₀` directivity (10.41) | `compute_pattern()` → `result["D0"]` (1D) |
| `D₀` directivity (10.40) | `compute_pattern_2d()` → `result["D0"]` (2D/3D) |
| Beam analysis | `analyze_cut()`, `analyze_pattern_1d()`, `analyze_pattern_2d()` in `analysis.py` |

## Performance Notes

- **NumPy vectorized**: fastest for all pipelines (~3x faster than pure C++, ~10x faster than pure Python)
- **Pure C++ pipeline** (`FullPipeline*`): 1.5–2.3x faster than C++ hybrid (eliminates ctypes marshalling)
- **C++ hybrid** (ctypes + NumPy): slowest C++ approach due to marshalling overhead
- C++ retains advantage for `ComplexArrayMagnitude` (29x vs Python loop) on `complex_t` structs

## Key Dependencies

Python: `numpy`, `matplotlib`, `PyQt6`, `PyYAML`, `ruff` (dev). See `requirements.txt`.
