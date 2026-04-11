"""Общие утилиты для расчёта и анализа антенных решёток."""

import csv
import logging
from pathlib import Path

import numpy as np
import yaml

SPEED_OF_LIGHT = 3e8
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def setup_logging():
    """Настройка логирования для всех модулей."""
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def load_config(section: str) -> dict:
    """Загружает секцию конфигурации из config.yaml."""
    with open(CONFIG_PATH) as f:
        return yaml.safe_load(f)[section]


def load_array_from_csv(path: Path) -> tuple[np.ndarray, ...]:
    """Загружает произвольное расположение элементов из CSV.

    Поддерживает три формата (заголовок необязателен, строки с # игнорируются):

    2 колонки (1D):  x_m, amplitude_db
    3 колонки (2D):  x_m, y_m, amplitude_db
    4 колонки (3D):  x_m, y_m, z_m, amplitude_db

    Возвращает:
        1D: (x_arr, amplitudes)
        2D: (x_arr, y_arr, amplitudes)
        3D: (x_arr, y_arr, z_arr, amplitudes)
    """
    rows = []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                vals = [float(v) for v in row]
            except ValueError:
                continue  # заголовок
            if len(vals) < 2:
                continue
            rows.append(vals)
    if not rows:
        raise ValueError(f"CSV файл пуст или не содержит числовых данных: {path}")

    ncols = len(rows[0])
    if ncols >= 4:
        x_arr = np.array([r[0] for r in rows])
        y_arr = np.array([r[1] for r in rows])
        z_arr = np.array([r[2] for r in rows])
        amplitudes = np.array([10 ** (r[3] / 20) for r in rows])
        return x_arr, y_arr, z_arr, amplitudes
    elif ncols >= 3:
        x_arr = np.array([r[0] for r in rows])
        y_arr = np.array([r[1] for r in rows])
        amplitudes = np.array([10 ** (r[2] / 20) for r in rows])
        return x_arr, y_arr, amplitudes
    else:
        x_arr = np.array([r[0] for r in rows])
        amplitudes = np.array([10 ** (r[1] / 20) for r in rows])
        return x_arr, amplitudes


def element_pattern(name: str, theta: np.ndarray) -> np.ndarray:
    """Нормированная амплитудная характеристика одного элемента f₁(θ).

    isotropic — изотропный излучатель: f₁ = 1
    cosine    — косинусная:            f₁ = |cos θ|
    dipole    — полуволновый вибратор: f₁ = |cos(π/2·sin θ) / cos θ|
    """
    if name == "isotropic":
        return np.ones_like(theta)
    if name == "cosine":
        return np.abs(np.cos(theta))
    if name == "dipole":
        cos_t = np.cos(theta)
        with np.errstate(divide="ignore", invalid="ignore"):
            f1 = np.where(
                np.abs(cos_t) < 1e-9,
                0.0,
                np.abs(np.cos(np.pi / 2 * np.sin(theta)) / cos_t),
            )
        return f1 / f1.max()
    raise ValueError(
        f"Unknown element_pattern: {name!r}. Use isotropic | cosine | dipole"
    )


def export_pattern_2d_csv(result: dict, path: Path, freq_hz: float = 0.0):
    """Экспорт 2D/3D ДН в CSV, совместимый с CST Studio Suite."""
    theta_deg = np.degrees(result["theta"])
    phi_deg = np.degrees(result["phi"])
    pattern_db = result["pattern_db"]

    with open(path, "w", newline="") as f:
        f.write("# Farfield Pattern Export\n")
        f.write(f"# Frequency [Hz]: {freq_hz:.6e}\n")
        f.write(f"# N elements: {result['N']}\n")
        f.write(f"# Directivity [dBi]: {result['D0_db']:.2f}\n")
        f.write(f"# Theta points: {len(theta_deg)}, Phi points: {len(phi_deg)}\n")
        f.write("Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]; Abs(F); Phase(F)[deg.]\n")
        af = result["array_factor"]
        phase = result["phase_deg"]
        for j, p in enumerate(phi_deg):
            for i, t in enumerate(theta_deg):
                f.write(
                    f"{t:.4f}; {p:.4f}; {pattern_db[i, j]:.4f}; "
                    f"{af[i, j]:.6f}; {phase[i, j]:.4f}\n"
                )
    logging.getLogger(__name__).info(
        "Exported pattern to %s (%d rows)", path, len(theta_deg) * len(phi_deg)
    )
