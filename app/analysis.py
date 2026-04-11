"""Анализ параметров главного лепестка диаграммы направленности.

Работает с данными в памяти (результаты compute_pattern / compute_pattern_2d / 3d)
и с экспортированными CSV-файлами (формат output_1d.csv / output_2d.csv).
"""

import csv
import logging
from pathlib import Path

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Вспомогательные функции
# ---------------------------------------------------------------------------


def _interp_crossing(x, y, level):
    """Находит точку пересечения уровня *level* линейной интерполяцией.

    x, y — два соседних отсчёта (x[0], y[0]) и (x[1], y[1]),
    между которыми y пересекает *level*.
    Возвращает x-координату пересечения.
    """
    dy = y[1] - y[0]
    if abs(dy) < 1e-15:
        return (x[0] + x[1]) / 2
    return x[0] + (level - y[0]) / dy * (x[1] - x[0])


def _find_level_crossing(theta_deg, pattern_db, start, direction, level):
    """Идёт от *start* в направлении *direction* (+1 / -1) и возвращает
    угол, при котором *pattern_db* пересекает *level* (дБ).

    Возвращает None, если пересечение не найдено.
    """
    n = len(pattern_db)
    i = start
    while 0 <= i + direction < n:
        j = i + direction
        if pattern_db[j] <= level:
            return _interp_crossing(
                (theta_deg[i], theta_deg[j]),
                (pattern_db[i], pattern_db[j]),
                level,
            )
        i = j
    return None


def _find_first_null(theta_deg, pattern_db, start, direction):
    """Находит первый локальный минимум (нуль ДН) от *start* в направлении *direction*.

    Возвращает (theta_deg, pattern_db) точки минимума или (None, None).
    """
    n = len(pattern_db)
    i = start
    # Сначала идём вниз, пока ДН убывает
    while 0 <= i + direction < n:
        j = i + direction
        if pattern_db[j] > pattern_db[i]:
            # i — локальный минимум
            return theta_deg[i], pattern_db[i]
        i = j
    return None, None


def _find_first_sidelobe(theta_deg, pattern_db, start, direction):
    """Находит первый боковой лепесток от *start* в направлении *direction*.

    Сначала ищет первый нуль, затем первый локальный максимум после него.
    Возвращает (theta_deg, level_db) вершины бокового лепестка или (None, None).
    """
    n = len(pattern_db)
    i = start

    # 1. Спуск до первого нуля (локального минимума)
    while 0 <= i + direction < n:
        j = i + direction
        if pattern_db[j] > pattern_db[i]:
            break  # i — нуль
        i = j
    else:
        return None, None

    # 2. Подъём до вершины бокового лепестка
    while 0 <= i + direction < n:
        j = i + direction
        if pattern_db[j] < pattern_db[i]:
            return theta_deg[i], pattern_db[i]
        i = j
    return None, None


# ---------------------------------------------------------------------------
# Основная функция анализа среза ДН
# ---------------------------------------------------------------------------


def analyze_cut(theta_deg: np.ndarray, pattern_db: np.ndarray) -> dict:
    """Определяет параметры главного лепестка по одномерному срезу ДН.

    Параметры:
        theta_deg   — угловые отсчёты, градусы (монотонно возрастающие)
        pattern_db  — ДН в дБ (нормированная: максимум ≈ 0 дБ)

    Возвращает словарь:
        peak_deg           — направление максимума, °
        beamwidth_3db      — ширина луча по −3 дБ, °  (None если не найдена)
        beamwidth_10db     — ширина луча по −10 дБ, ° (None если не найдена)
        first_null_left_deg  — первый нуль слева, °
        first_null_right_deg — первый нуль справа, °
        first_sll_db       — УБЛ первого бокового лепестка, дБ (наихудший из двух сторон)
        first_sll_left_db  — УБЛ слева, дБ
        first_sll_right_db — УБЛ справа, дБ
    """
    peak_idx = int(np.argmax(pattern_db))
    peak_deg = float(theta_deg[peak_idx])
    peak_db = float(pattern_db[peak_idx])

    # Ширина луча по −3 дБ
    level_3 = peak_db - 3.0
    left_3 = _find_level_crossing(theta_deg, pattern_db, peak_idx, -1, level_3)
    right_3 = _find_level_crossing(theta_deg, pattern_db, peak_idx, +1, level_3)
    bw_3 = (right_3 - left_3) if (left_3 is not None and right_3 is not None) else None

    # Ширина луча по −10 дБ
    level_10 = peak_db - 10.0
    left_10 = _find_level_crossing(theta_deg, pattern_db, peak_idx, -1, level_10)
    right_10 = _find_level_crossing(theta_deg, pattern_db, peak_idx, +1, level_10)
    bw_10 = (right_10 - left_10) if (left_10 is not None and right_10 is not None) else None

    # Первые нули
    null_left_deg, _ = _find_first_null(theta_deg, pattern_db, peak_idx, -1)
    null_right_deg, _ = _find_first_null(theta_deg, pattern_db, peak_idx, +1)

    # Первые боковые лепестки
    _, sll_left_db = _find_first_sidelobe(theta_deg, pattern_db, peak_idx, -1)
    _, sll_right_db = _find_first_sidelobe(theta_deg, pattern_db, peak_idx, +1)

    # УБЛ — наихудший (наибольший) из двух сторон
    sll_values = [v for v in (sll_left_db, sll_right_db) if v is not None]
    first_sll_db = max(sll_values) if sll_values else None

    # Сектор главного лепестка (ширина по нулям)
    if null_left_deg is not None and null_right_deg is not None:
        sector_deg = null_right_deg - null_left_deg
    else:
        sector_deg = None

    # Симметрия относительно пика: левая/правая полуширина
    half_left_3 = (peak_deg - left_3) if left_3 is not None else None
    half_right_3 = (right_3 - peak_deg) if right_3 is not None else None
    if half_left_3 is not None and half_right_3 is not None and half_right_3 > 0:
        symmetry_3db = half_left_3 / half_right_3
    else:
        symmetry_3db = None

    half_left_null = (peak_deg - null_left_deg) if null_left_deg is not None else None
    half_right_null = (null_right_deg - peak_deg) if null_right_deg is not None else None
    if half_left_null is not None and half_right_null is not None and half_right_null > 0:
        symmetry_null = half_left_null / half_right_null
    else:
        symmetry_null = None

    return {
        "peak_deg": peak_deg,
        "beamwidth_3db": bw_3,
        "beamwidth_10db": bw_10,
        "sector_deg": sector_deg,
        "first_null_left_deg": null_left_deg,
        "first_null_right_deg": null_right_deg,
        "first_sll_db": first_sll_db,
        "first_sll_left_db": sll_left_db,
        "first_sll_right_db": sll_right_db,
        "symmetry_3db": symmetry_3db,
        "symmetry_null": symmetry_null,
    }


# ---------------------------------------------------------------------------
# Обёртки для результатов compute_pattern_*
# ---------------------------------------------------------------------------


def analyze_pattern_1d(result: dict) -> dict:
    """Анализ результата compute_pattern() (1D).

    Возвращает словарь с параметрами главного лепестка + D0/D0_db.
    """
    cut = analyze_cut(result["theta_deg"], result["pattern_db"])
    cut["D0"] = result.get("D0")
    cut["D0_db"] = result.get("D0_db")
    return cut


def analyze_pattern_2d(result: dict) -> dict:
    """Анализ результата compute_pattern_2d() / compute_pattern_3d() (2D).

    Извлекает срезы φ=0° (xz) и φ=90° (yz), анализирует каждый.
    Возвращает словарь с ключами cut_xz, cut_yz, D0, D0_db.
    """
    theta = result["theta"]  # радианы
    phi = result["phi"]  # радианы
    pattern_db = result["pattern_db"]  # [n_theta, n_phi]

    theta_deg = np.degrees(theta)

    i_phi_0 = int(np.argmin(np.abs(phi - 0.0)))
    i_phi_90 = int(np.argmin(np.abs(phi - np.pi / 2)))

    cut_xz = analyze_cut(theta_deg, pattern_db[:, i_phi_0])
    cut_yz = analyze_cut(theta_deg, pattern_db[:, i_phi_90])

    # Пространственный сектор главного лепестка
    # По −3 дБ: Ω₃ ≈ Δθ_xz · Δθ_yz (стерадианы, для малых углов)
    bw_xz = cut_xz["beamwidth_3db"]
    bw_yz = cut_yz["beamwidth_3db"]
    if bw_xz is not None and bw_yz is not None:
        beam_solid_angle_3db_sr = np.radians(bw_xz) * np.radians(bw_yz)
    else:
        beam_solid_angle_3db_sr = None

    # По нулям: Ω₀ ≈ sector_xz · sector_yz
    sec_xz = cut_xz["sector_deg"]
    sec_yz = cut_yz["sector_deg"]
    if sec_xz is not None and sec_yz is not None:
        beam_solid_angle_null_sr = np.radians(sec_xz) * np.radians(sec_yz)
    else:
        beam_solid_angle_null_sr = None

    # Коэффициент эллиптичности: отношение ширины луча в двух плоскостях
    # ellipticity = 1.0 для круглого (квадратная решётка), >1 для вытянутого
    if bw_xz is not None and bw_yz is not None and bw_yz > 0:
        ellipticity = max(bw_xz, bw_yz) / min(bw_xz, bw_yz)
    else:
        ellipticity = None

    return {
        "cut_xz": cut_xz,
        "cut_yz": cut_yz,
        "beam_solid_angle_3db_sr": beam_solid_angle_3db_sr,
        "beam_solid_angle_null_sr": beam_solid_angle_null_sr,
        "ellipticity": ellipticity,
        "D0": result.get("D0"),
        "D0_db": result.get("D0_db"),
    }


# ---------------------------------------------------------------------------
# Загрузка и анализ CSV
# ---------------------------------------------------------------------------


def analyze_csv(path: Path) -> dict:
    """Загружает экспортированный CSV и анализирует ДН.

    Формат CSV (разделитель — точка с запятой):
        Theta [deg.]; Phi [deg.]; Abs(Dir.)[dB]; ...

    Строки с # — комментарии. Первая строка данных — заголовок колонок.
    Для 1D (один φ) возвращает результат analyze_cut.
    Для 2D (несколько φ) возвращает результат с cut_xz и cut_yz.
    """
    thetas, phis, pattern = [], [], []
    with open(path, newline="") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            if line.startswith("Theta"):
                continue  # заголовок колонок
            parts = line.split(";")
            if len(parts) < 3:
                continue
            try:
                thetas.append(float(parts[0]))
                phis.append(float(parts[1]))
                pattern.append(float(parts[2]))
            except ValueError:
                continue

    thetas = np.array(thetas)
    phis = np.array(phis)
    pattern = np.array(pattern)

    unique_phi = np.unique(phis)

    if len(unique_phi) <= 1:
        # 1D: один срез
        return analyze_cut(thetas, pattern)

    # 2D: восстанавливаем сетку
    unique_theta = np.unique(thetas)
    n_theta = len(unique_theta)
    n_phi = len(unique_phi)

    # CSV записан как: для каждого phi перебираются все theta
    pattern_2d = pattern.reshape(n_phi, n_theta).T  # → [n_theta, n_phi]

    i_phi_0 = int(np.argmin(np.abs(unique_phi - 0.0)))
    i_phi_90 = int(np.argmin(np.abs(unique_phi - 90.0)))

    cut_xz = analyze_cut(unique_theta, pattern_2d[:, i_phi_0])
    cut_yz = analyze_cut(unique_theta, pattern_2d[:, i_phi_90])

    return {
        "cut_xz": cut_xz,
        "cut_yz": cut_yz,
    }


# ---------------------------------------------------------------------------
# Форматированный вывод
# ---------------------------------------------------------------------------


def format_analysis(analysis: dict) -> str:
    """Форматирует результат анализа в читаемую строку."""
    lines = []

    if "cut_xz" in analysis:
        # 2D результат
        if analysis.get("D0_db") is not None:
            lines.append(f"  КНД D₀ = {analysis['D0']:.2f} ({analysis['D0_db']:.2f} дБ)")
        for name, key in [("φ=0° (xz)", "cut_xz"), ("φ=90° (yz)", "cut_yz")]:
            cut = analysis[key]
            lines.append(f"  Срез {name}:")
            lines.append(f"    Максимум: {cut['peak_deg']:.2f}°")
            if cut["beamwidth_3db"] is not None:
                lines.append(f"    Ширина луча (−3 дБ):  {cut['beamwidth_3db']:.2f}°")
            if cut["beamwidth_10db"] is not None:
                lines.append(f"    Ширина луча (−10 дБ): {cut['beamwidth_10db']:.2f}°")
            if cut["sector_deg"] is not None:
                lines.append(f"    Сектор по нулям:      {cut['sector_deg']:.2f}°")
            nulls = []
            if cut["first_null_left_deg"] is not None:
                nulls.append(f"{cut['first_null_left_deg']:.2f}°")
            if cut["first_null_right_deg"] is not None:
                nulls.append(f"{cut['first_null_right_deg']:.2f}°")
            if nulls:
                lines.append(f"    Первые нули: {', '.join(nulls)}")
            if cut["first_sll_db"] is not None:
                lines.append(f"    УБЛ: {cut['first_sll_db']:.2f} дБ")
            if cut["symmetry_3db"] is not None:
                lines.append(f"    Симметрия (−3 дБ):    {cut['symmetry_3db']:.3f}")
            if cut["symmetry_null"] is not None:
                lines.append(f"    Симметрия (нули):     {cut['symmetry_null']:.3f}")
        if analysis.get("ellipticity") is not None:
            lines.append(f"  Коэффициент эллиптичности: {analysis['ellipticity']:.3f}")
        if analysis.get("beam_solid_angle_3db_sr") is not None:
            lines.append(
                f"  Пространственный сектор (−3 дБ):  {analysis['beam_solid_angle_3db_sr']:.4f} ср"
            )
        if analysis.get("beam_solid_angle_null_sr") is not None:
            lines.append(
                f"  Пространственный сектор (нули):   {analysis['beam_solid_angle_null_sr']:.4f} ср"
            )
    else:
        # 1D результат
        if analysis.get("D0_db") is not None:
            lines.append(f"  КНД D₀ = {analysis['D0']:.2f} ({analysis['D0_db']:.2f} дБ)")
        lines.append(f"  Максимум: {analysis['peak_deg']:.2f}°")
        if analysis["beamwidth_3db"] is not None:
            lines.append(f"  Ширина луча (−3 дБ):  {analysis['beamwidth_3db']:.2f}°")
        if analysis["beamwidth_10db"] is not None:
            lines.append(f"  Ширина луча (−10 дБ): {analysis['beamwidth_10db']:.2f}°")
        if analysis.get("sector_deg") is not None:
            lines.append(f"  Сектор по нулям:      {analysis['sector_deg']:.2f}°")
        nulls = []
        if analysis["first_null_left_deg"] is not None:
            nulls.append(f"{analysis['first_null_left_deg']:.2f}°")
        if analysis["first_null_right_deg"] is not None:
            nulls.append(f"{analysis['first_null_right_deg']:.2f}°")
        if nulls:
            lines.append(f"  Первые нули: {', '.join(nulls)}")
        if analysis["first_sll_db"] is not None:
            lines.append(f"  УБЛ: {analysis['first_sll_db']:.2f} дБ")
        if analysis.get("symmetry_3db") is not None:
            lines.append(f"  Симметрия (−3 дБ):    {analysis['symmetry_3db']:.3f}")
        if analysis.get("symmetry_null") is not None:
            lines.append(f"  Симметрия (нули):     {analysis['symmetry_null']:.3f}")

    return "\n".join(lines)
