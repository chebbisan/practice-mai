import ctypes as ct
import csv
import logging
import platform
from pathlib import Path
from typing import TypedDict

import yaml

import matplotlib.pyplot as plt
import numpy as np

from complex import complex_t
from util import (
    initialize_library,
    list_to_c_double_array,
    list_to_c_complex_array,
    calculate_delta_x,
)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)

SPEED_OF_LIGHT = 3 * 10**8

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def load_array_from_csv(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Загружает произвольное расположение элементов из CSV.

    Формат файла (заголовок необязателен, строки с # игнорируются):
        x_m,amplitude_db
        0.00,0.0
        0.05,-3.0

    Возвращает:
        x_arr      — координаты элементов, м
        amplitudes — амплитуды в линейном масштабе (из дБ: A = 10^(dB/20))
    """
    x_list, amp_list = [], []
    with open(path, newline="") as f:
        reader = csv.reader(f)
        for row in reader:
            if not row or row[0].strip().startswith("#"):
                continue
            try:
                x = float(row[0])
                amp_db = float(row[1])
            except ValueError:
                continue  # заголовок
            x_list.append(x)
            amp_list.append(10 ** (amp_db / 20))
    if not x_list:
        raise ValueError(f"CSV файл пуст или не содержит числовых данных: {path}")
    return np.array(x_list), np.array(amp_list)


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
    raise ValueError(f"Unknown element_pattern: {name!r}. Use isotropic | cosine | dipole")


# ---------------------------------------------------------------------------
# Класс результата
# ---------------------------------------------------------------------------


class PatternResult(TypedDict):
    theta_deg:    np.ndarray   # углы, градусы
    pattern_db:   np.ndarray   # полная нормированная ДН, дБ
    array_factor: np.ndarray   # только множитель решётки (линейный, норм.)
    f1:           np.ndarray   # характеристика элемента (линейный, норм.)
    N:            int          # число элементов
    label:        str          # описание для легенды


# ---------------------------------------------------------------------------
# 1. Функция расчёта (без GUI, без I/O)
# ---------------------------------------------------------------------------


def compute_pattern(
    x_arr: np.ndarray,
    amplitudes: np.ndarray,
    freq_hz: float,
    n_theta: int,
    elem_pattern_name: str,
    c_lib,
) -> PatternResult:
    """Рассчитывает ДН линейной АР с произвольным расположением элементов.

    Параметры:
        x_arr            — координаты элементов, м  (форма [N])
        amplitudes       — линейные амплитуды возбуждения [N]  (≥ 0)
        freq_hz          — несущая частота, Гц
        n_theta          — число точек по углу
        elem_pattern_name — 'isotropic' | 'cosine' | 'dipole'
        c_lib            — загруженная C-библиотека (initialize_library)

    Возвращает PatternResult со всеми промежуточными данными.
    """
    N = len(x_arr)
    wave_length = SPEED_OF_LIGHT / freq_hz
    wave_num = 2 * np.pi / wave_length
    logger.debug("compute_pattern: N=%d, λ=%.4f m, k=%.4f rad/m", N, wave_length, wave_num)

    theta = np.linspace(-np.pi / 2, np.pi / 2, n_theta)

    f_arr = [complex_t(float(a), 0.0) for a in amplitudes]
    c_f = list_to_c_complex_array(f_arr)
    c_x = list_to_c_double_array(x_arr)
    c_theta = list_to_c_double_array(theta)

    raw = c_lib.Calculate1DAntennaArray(
        ct.c_int(N), ct.c_int(n_theta),
        c_f, c_x, c_theta, ct.c_double(wave_num),
    )
    af = np.array([abs(complex(raw[i].real, raw[i].imag)) for i in range(n_theta)])
    c_lib.FreeComplexArr(raw)

    f1 = element_pattern(elem_pattern_name, theta)

    full = f1 * af
    full /= full.max()

    return PatternResult(
        theta_deg=np.degrees(theta),
        pattern_db=20 * np.log10(np.maximum(full, 1e-10)),
        array_factor=af / af.max(),
        f1=f1,
        N=N,
        label=f"N={N}, эл-т: {elem_pattern_name}",
    )


# ---------------------------------------------------------------------------
# 2. Функция отображения (только график)
# ---------------------------------------------------------------------------


def plot_pattern(result: PatternResult, title: str = "Диаграмма направленности линейной АР"):
    """Строит график ДН по результату compute_pattern.

    Параметры:
        result — PatternResult, возвращённый compute_pattern
        title  — заголовок графика
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(result["theta_deg"], result["pattern_db"], label=result["label"])
    ax.axhline(-3,  color="red",   linestyle="--", label="-3 дБ")
    ax.axhline(-13, color="green", linestyle="--", label="-13 дБ")
    ax.set_xlabel(r"$\theta$, градус")
    ax.set_ylabel(r"$|F(\theta)|$, дБ", rotation=0)
    ax.set_title(title)
    ax.legend()
    ax.grid(True)
    plt.tight_layout()
    plt.show()


# ---------------------------------------------------------------------------
# Точка входа (скрипт)
# ---------------------------------------------------------------------------


def main():
    logger.info("Starting 1D antenna array calculation")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["array_1d"]

    freq_0            = cfg["freq_hz"]
    n_theta           = cfg["n_theta"]
    elem_pattern_name = cfg.get("element_pattern", "isotropic")
    csv_file          = cfg.get("csv_file")

    c_lib = initialize_library(str(LIB_PATH))
    wave_length = SPEED_OF_LIGHT / freq_0

    if csv_file is not None:
        csv_path = Path(csv_file) if Path(csv_file).is_absolute() \
            else Path(__file__).parent / csv_file
        x_arr, amplitudes = load_array_from_csv(csv_path)
        extra_label = f"CSV: {csv_path.name}, "
        logger.info("CSV mode: %s, N=%d", csv_path, len(x_arr))
    else:
        N         = cfg["N"]
        d         = cfg["d"]
        steer_deg = cfg["steer_deg"]
        delta_x = d if d is not None else calculate_delta_x(wave_length, np.radians(steer_deg))
        L = delta_x * (N - 1)
        x_arr = np.array([i * delta_x - L / 2 for i in range(N)])
        amplitudes = np.ones(N)
        extra_label = f"d={delta_x:.4f} м, "
        logger.debug("Uniform mode: N=%d, d=%.4f m", N, delta_x)

    result = compute_pattern(x_arr, amplitudes, freq_0, n_theta, elem_pattern_name, c_lib)
    result["label"] = extra_label + result["label"]

    plot_pattern(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
