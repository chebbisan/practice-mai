import logging
from pathlib import Path
from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np

from common import (
    SPEED_OF_LIGHT,
    setup_logging,
    load_config,
    load_array_from_csv,
    element_pattern,
)
from analysis import analyze_pattern_1d, format_analysis
from util import calculate_delta_x

setup_logging()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Класс результата
# ---------------------------------------------------------------------------


class PatternResult(TypedDict):
    theta_deg: np.ndarray  # углы, градусы
    pattern_db: np.ndarray  # полная нормированная ДН, дБ
    array_factor: np.ndarray  # только множитель решётки (линейный, норм.)
    phase_deg: np.ndarray  # фаза множителя решётки, градусы
    f1: np.ndarray  # характеристика элемента (линейный, норм.)
    N: int  # число элементов
    D0: float  # КНД в направлении максимума (безразм.)
    D0_db: float  # КНД в дБ
    label: str  # описание для легенды


# ---------------------------------------------------------------------------
# 1. Функция расчёта (NumPy vectorized)
# ---------------------------------------------------------------------------


def compute_pattern(
    x_arr: np.ndarray,
    amplitudes: np.ndarray,
    freq_hz: float,
    n_theta: int,
    elem_pattern_name: str,
) -> PatternResult:
    """Рассчитывает ДН линейной АР с произвольным расположением элементов."""
    N = len(x_arr)
    wave_length = SPEED_OF_LIGHT / freq_hz
    wave_num = 2 * np.pi / wave_length
    logger.debug(
        "compute_pattern: N=%d, λ=%.4f m, k=%.4f rad/m", N, wave_length, wave_num
    )

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)

    # NumPy vectorized: AF(θ) = (1/N) Σ aₙ exp(-jk xₙ sinθ)
    phase = -wave_num * np.outer(x_arr, np.sin(theta))
    af_complex = np.sum(amplitudes[:, np.newaxis] * np.exp(1j * phase), axis=0) / N

    af = np.abs(af_complex)
    f1 = element_pattern(elem_pattern_name, theta)

    full = f1 * af
    full /= full.max()
    phase_deg = np.degrees(np.angle(af_complex))

    # КНД по формуле 10.41 (осевая симметрия): D₀ = 2 / ∫ F²(Θ)cosΘ dΘ
    D0 = 2.0 / np.trapezoid(full**2 * np.cos(theta), theta)
    D0_db = 10 * np.log10(D0)
    logger.debug("Directivity D0=%.2f (%.2f dB)", D0, D0_db)

    return PatternResult(
        theta_deg=np.degrees(theta),
        pattern_db=20 * np.log10(np.maximum(full, 1e-10)),
        array_factor=af / af.max(),
        phase_deg=phase_deg,
        f1=f1,
        N=N,
        D0=D0,
        D0_db=D0_db,
        label=f"N={N}, эл-т: {elem_pattern_name}",
    )


# ---------------------------------------------------------------------------
# 2. Визуализация
# ---------------------------------------------------------------------------


def plot_array_1d(x_arr: np.ndarray, amplitudes: np.ndarray):
    """Отображает расположение элементов линейной АР."""
    fig, ax = plt.subplots(figsize=(8, 2.5))
    amp_db = 20 * np.log10(np.maximum(amplitudes / amplitudes.max(), 1e-10))
    sc = ax.scatter(
        x_arr * 1e3,
        np.zeros_like(x_arr),
        c=amp_db,
        cmap="viridis",
        s=120,
        edgecolors="black",
        vmin=-20,
        vmax=0,
        zorder=3,
    )
    ax.set_xlabel("x, мм")
    ax.set_yticks([])
    ax.set_title(f"Расположение элементов (N={len(x_arr)})")
    fig.colorbar(
        sc, ax=ax, label="Амплитуда, дБ", orientation="horizontal", pad=0.25, aspect=40
    )
    ax.grid(True, axis="x")
    plt.tight_layout()
    plt.show()


def plot_pattern(
    result: PatternResult, title: str = "Диаграмма направленности линейной АР"
):
    """Строит график ДН по результату compute_pattern."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result["theta_deg"], result["pattern_db"], label=result["label"])

    ax.set_xlabel(r"$\theta$, градус")
    ax.set_ylabel(r"$|F(\theta)|$, дБ", rotation=0)
    ax.set_ylim(-60, 0)
    ax.set_title(f"{title}\nКНД $D_0$ = {result['D0']:.2f} ({result['D0_db']:.2f} дБ)")
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# 3. Экспорт в CST-совместимый формат
# ---------------------------------------------------------------------------


def export_pattern_csv(result: PatternResult, path: Path, freq_hz: float = 0.0):
    """Экспорт 1D ДН в CSV, совместимый с CST Studio Suite."""
    with open(path, "w", newline="") as f:
        f.write("# Farfield Pattern Export\n")
        f.write(f"# Frequency [Hz]: {freq_hz:.6e}\n")
        f.write(f"# N elements: {result['N']}\n")
        f.write(f"# Directivity [dBi]: {result['D0_db']:.2f}\n")
        f.write("Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]; Abs(F); Phase(F)[deg.]\n")
        af = result["array_factor"]
        phase = result["phase_deg"]
        for i, theta_deg in enumerate(result["theta_deg"]):
            f.write(
                f"{theta_deg:.4f}; 0.0000; {result['pattern_db'][i]:.4f}; "
                f"{af[i]:.6f}; {phase[i]:.4f}\n"
            )
    logger.info("Exported 1D pattern to %s", path)


# ---------------------------------------------------------------------------
# Точка входа
# ---------------------------------------------------------------------------


def main():
    logger.info("Starting 1D antenna array calculation")

    cfg = load_config("array_1d")
    freq_0 = cfg["freq_hz"]
    n_theta = cfg["n_theta"]
    elem_pattern_name = cfg.get("element_pattern", "isotropic")
    csv_file = cfg.get("csv_file")

    wave_length = SPEED_OF_LIGHT / freq_0

    if csv_file is not None:
        csv_path = (
            Path(csv_file)
            if Path(csv_file).is_absolute()
            else Path(__file__).parent / csv_file
        )
        x_arr, amplitudes = load_array_from_csv(csv_path)
        extra_label = f"CSV: {csv_path.name}, "
        logger.info("CSV mode: %s, N=%d", csv_path, len(x_arr))
    else:
        N = cfg["N"]
        d = cfg["d"]
        steer_deg = cfg["steer_deg"]
        delta_x = (
            d
            if d is not None
            else calculate_delta_x(wave_length, np.radians(steer_deg))
        )
        L = delta_x * (N - 1)
        x_arr = np.array([i * delta_x - L / 2 for i in range(N)])
        amplitudes = np.ones(N)
        extra_label = f"d={delta_x:.4f} м, "
        logger.debug("Uniform mode: N=%d, d=%.4f m", N, delta_x)

    result = compute_pattern(x_arr, amplitudes, freq_0, n_theta, elem_pattern_name)
    result["label"] = extra_label + result["label"]

    export_pattern_csv(
        result, Path(__file__).parent / "output" / "output_1d.csv", freq_0
    )

    analysis = analyze_pattern_1d(result)
    logger.info("Параметры главного лепестка:\n%s", format_analysis(analysis))

    plot_array_1d(x_arr, amplitudes)
    plot_pattern(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
