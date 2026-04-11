import logging
from pathlib import Path

import yaml

import matplotlib.pyplot as plt
import numpy as np

from calc_1d import load_array_from_csv, element_pattern
from plot_2d import plot_heatmap

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 3 * 10**8

CONFIG_PATH = Path(__file__).parent / "config.yaml"


def compute_pattern_2d(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    amplitudes: np.ndarray,
    freq_hz: float,
    n_theta: int,
    n_phi: int,
    elem_pattern_name: str,
) -> dict:
    N = len(x_arr)
    wave_length = SPEED_OF_LIGHT / freq_hz
    wave_num = 2 * np.pi / wave_length
    logger.debug(
        "compute_pattern_2d: N=%d, lambda=%.4f m, k=%.4f rad/m",
        N,
        wave_length,
        wave_num,
    )

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    phi = np.linspace(-np.pi / 2, np.pi / 2, n_phi)

    # NumPy vectorized array factor: AF(θ,φ) = (1/N) Σ aₙ exp(-jk(xₙsinθcosφ + yₙsinθsinφ))
    st_cp = np.outer(np.sin(theta), np.cos(phi))  # [n_theta, n_phi]
    st_sp = np.outer(np.sin(theta), np.sin(phi))  # [n_theta, n_phi]
    af_complex = np.zeros((n_theta, n_phi), dtype=complex)
    for n in range(N):
        af_complex += amplitudes[n] * np.exp(
            -1j * wave_num * (x_arr[n] * st_cp + y_arr[n] * st_sp)
        )
    af_complex /= N

    af_2d = np.abs(af_complex)
    phase_2d = np.degrees(np.angle(af_complex))

    f1_theta = element_pattern(elem_pattern_name, theta)
    f1_2d = f1_theta[:, np.newaxis] * np.ones((1, n_phi))

    full = f1_2d * af_2d
    full /= full.max()

    pattern_db = 20 * np.log10(np.maximum(full, 1e-10))

    peak_idx = np.unravel_index(np.argmax(full), full.shape)

    # КНД по формуле 10.40: D₀ = 4π / ∫∫ F²(Θ,φ) cosΘ dΘ dφ
    integrand = full**2 * np.cos(theta)[:, np.newaxis]
    inner = np.trapezoid(integrand, theta, axis=0)
    D0 = 4 * np.pi / np.trapezoid(inner, phi)
    D0_db = 10 * np.log10(D0)
    logger.debug("Directivity D0=%.2f (%.2f dB)", D0, D0_db)

    return {
        "theta": theta,
        "phi": phi,
        "pattern_2d": full,
        "pattern_db": pattern_db,
        "array_factor": af_2d / af_2d.max(),
        "phase_deg": phase_2d,
        "f1_2d": f1_2d,
        "N": N,
        "D0": D0,
        "D0_db": D0_db,
        "peak_theta_idx": peak_idx[0],
        "peak_phi_idx": peak_idx[1],
    }


def plot_array_2d(x_arr: np.ndarray, y_arr: np.ndarray, amplitudes: np.ndarray):
    """Отображает расположение элементов 2D АР."""
    fig, ax = plt.subplots(figsize=(6, 6))
    amp_db = 20 * np.log10(np.maximum(amplitudes / amplitudes.max(), 1e-10))
    sc = ax.scatter(
        x_arr * 1e3,
        y_arr * 1e3,
        c=amp_db,
        cmap="viridis",
        s=120,
        edgecolors="black",
        vmin=-20,
        vmax=0,
        zorder=3,
    )
    ax.set_xlabel("x, мм")
    ax.set_ylabel("y, мм")
    ax.set_title(f"Расположение элементов (N={len(x_arr)})")
    ax.set_aspect("equal")
    fig.colorbar(sc, ax=ax, label="Амплитуда, дБ")
    ax.grid(True)
    plt.tight_layout()
    plt.show()


def export_pattern_2d_csv(result: dict, path: Path, freq_hz: float = 0.0):
    """Экспорт 2D ДН в CSV, совместимый с CST Studio Suite.

    Формат: Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]
    Данные записываются по строкам: для каждого Phi перебираются все Theta.
    """
    theta_deg = np.degrees(result["theta"])
    phi_deg = np.degrees(result["phi"])
    pattern_db = result["pattern_db"]

    with open(path, "w", newline="") as f:
        f.write(f"# Farfield Pattern Export\n")
        f.write(f"# Frequency [Hz]: {freq_hz:.6e}\n")
        f.write(f"# N elements: {result['N']}\n")
        f.write(f"# Directivity [dBi]: {result['D0_db']:.2f}\n")
        f.write(f"# Theta points: {len(theta_deg)}, Phi points: {len(phi_deg)}\n")
        f.write(
            "Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]; Abs(F); Phase(F)[deg.]\n"
        )
        af = result["array_factor"]
        phase = result["phase_deg"]
        for j, p in enumerate(phi_deg):
            for i, t in enumerate(theta_deg):
                f.write(
                    f"{t:.4f}; {p:.4f}; {pattern_db[i, j]:.4f}; "
                    f"{af[i, j]:.6f}; {phase[i, j]:.4f}\n"
                )
    logger.info("Exported 2D pattern to %s (%d rows)", path, len(theta_deg) * len(phi_deg))


def plot_2d_with_cuts(result: dict):
    theta = result["theta"]
    phi = result["phi"]
    pattern_db = result["pattern_db"]

    plot_heatmap(theta, phi, pattern_db)

    # Срезы в главных плоскостях: φ=0° (xz) и φ=90° (yz)
    i_phi_0 = np.argmin(np.abs(phi - 0.0))
    i_phi_90 = np.argmin(np.abs(phi - np.pi / 2))
    cut_xz = pattern_db[:, i_phi_0]
    cut_yz = pattern_db[:, i_phi_90]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    fig.suptitle(f"КНД $D_0$ = {result['D0']:.2f} ({result['D0_db']:.2f} дБ)")

    ax1.plot(np.degrees(theta), cut_xz)
    ax1.set_xlabel(r"$\theta$, deg")
    ax1.set_ylabel(r"$|F|$, dB")
    ax1.set_ylim(-60, 0)
    ax1.set_title(r"$\varphi=0°$ (xz, вдоль x)")
    ax1.grid(True)

    ax2.plot(np.degrees(theta), cut_yz)
    ax2.set_xlabel(r"$\theta$, deg")
    ax2.set_ylabel(r"$|F|$, dB")
    ax2.set_ylim(-60, 0)
    ax2.set_title(r"$\varphi=90°$ (yz, вдоль y)")
    ax2.grid(True)

    plt.tight_layout()
    plt.show()


def main():
    logger.info("Starting 2D antenna array calculation")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["array_2d"]

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
    if len(data) != 3:
        raise ValueError("2D array CSV must have 3 columns: x_m, y_m, amplitude_db")
    x_arr, y_arr, amplitudes = data

    logger.info("Loaded %d elements from %s", len(x_arr), csv_path)

    result = compute_pattern_2d(
        x_arr,
        y_arr,
        amplitudes,
        freq_hz,
        n_theta,
        n_phi,
        elem_pattern_name,
    )

    logger.info(
        "N=%d, peak at theta[%d]=%.1f deg, phi[%d]=%.1f deg",
        result["N"],
        result["peak_theta_idx"],
        np.degrees(result["theta"][result["peak_theta_idx"]]),
        result["peak_phi_idx"],
        np.degrees(result["phi"][result["peak_phi_idx"]]),
    )

    export_pattern_2d_csv(result, Path(__file__).parent / "output" / "output_2d.csv", freq_hz)

    plot_array_2d(x_arr, y_arr, amplitudes)
    plot_2d_with_cuts(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
