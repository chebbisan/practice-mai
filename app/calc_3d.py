import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from common import (
    setup_logging,
    load_config,
    load_array_from_csv,
    export_pattern_2d_csv,
)
from analysis import analyze_pattern_2d, format_analysis
from calc_2d import compute_pattern_2d, plot_cuts

setup_logging()
logger = logging.getLogger(__name__)


def compute_pattern_3d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    z_arr: np.ndarray,
    amplitudes: np.ndarray,
    freq_hz: float,
    n_theta: int,
    n_phi: int,
    elem_pattern_name: str,
) -> dict:
    """Рассчитывает ДН пространственной (3D) АР. Делегирует в compute_pattern_2d."""
    return compute_pattern_2d(
        x_arr,
        y_arr,
        amplitudes,
        freq_hz,
        n_theta,
        n_phi,
        elem_pattern_name,
        z_arr=z_arr,
    )


def plot_array_3d(
    x_arr: np.ndarray, y_arr: np.ndarray, z_arr: np.ndarray, amplitudes: np.ndarray
):
    """Отображает расположение элементов 3D АР."""
    fig = plt.figure(figsize=(8, 6))
    ax = fig.add_subplot(111, projection="3d")
    amp_db = 20 * np.log10(np.maximum(amplitudes / amplitudes.max(), 1e-10))
    sc = ax.scatter(
        x_arr * 1e3,
        y_arr * 1e3,
        z_arr * 1e3,
        c=amp_db,
        cmap="viridis",
        s=120,
        edgecolors="black",
        vmin=-20,
        vmax=0,
    )
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_zlabel("z, мм")
    ax.set_title(f"Расположение элементов (N={len(x_arr)})")
    fig.colorbar(sc, ax=ax, label="Амплитуда, дБ", shrink=0.6)
    plt.tight_layout()
    plt.show()


def main():
    logger.info("Starting 3D antenna array calculation")

    cfg = load_config("array_3d")
    freq_hz = cfg["freq_hz"]
    n_theta = cfg["n_theta"]
    n_phi = cfg["n_phi"]
    elem_pattern_name = cfg.get("element_pattern", "isotropic")
    csv_file = cfg["csv_file"]

    csv_path = (
        Path(csv_file)
        if Path(csv_file).is_absolute()
        else Path(__file__).parent / csv_file
    )

    data = load_array_from_csv(csv_path)
    if len(data) != 4:
        raise ValueError(
            "3D array CSV must have 4 columns: x_m, y_m, z_m, amplitude_db"
        )
    x_arr, y_arr, z_arr, amplitudes = data

    logger.info("Loaded %d elements from %s", len(x_arr), csv_path)

    result = compute_pattern_3d(
        x_arr, y_arr, z_arr, amplitudes, freq_hz, n_theta, n_phi, elem_pattern_name
    )

    logger.info(
        "N=%d, peak at theta[%d]=%.1f deg, phi[%d]=%.1f deg",
        result["N"],
        result["peak_theta_idx"],
        np.degrees(result["theta"][result["peak_theta_idx"]]),
        result["peak_phi_idx"],
        np.degrees(result["phi"][result["peak_phi_idx"]]),
    )

    export_pattern_2d_csv(
        result, Path(__file__).parent / "output" / "output_3d.csv", freq_hz
    )

    analysis = analyze_pattern_2d(result)
    logger.info("Параметры главного лепестка:\n%s", format_analysis(analysis))

    plot_array_3d(x_arr, y_arr, z_arr, amplitudes)
    plot_cuts(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
