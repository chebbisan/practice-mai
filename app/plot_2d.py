import logging

import matplotlib.pyplot as plt
import numpy as np

logger = logging.getLogger(__name__)


def plot_heatmap(theta, phi, log_pattern):
    """Круговая тепловая карта ДН в полярных координатах.

    Входные данные: θ ∈ [-π/2, π/2], φ ∈ [-π/2, π/2].
    AF(-θ, φ) = AF(θ, φ+π), поэтому берём θ ≥ 0 и расширяем φ до полного круга.
    Проекция: u = sinθ·cosφ, v = sinθ·sinφ.
    """
    logger.info("Plotting heatmap (polar)")
    n_theta = len(theta)
    mid = n_theta // 2  # индекс θ=0

    # Верхняя полусфера (θ ≥ 0): φ как есть
    theta_pos = theta[mid:]  # включает θ=0
    pat_pos = log_pattern[mid:, :]

    # Нижняя → отражение: θ < 0 → (|θ|, φ+π); пропускаем θ=0 (уже в pos)
    theta_neg = -theta[:mid][::-1]  # |θ| по возрастанию, без 0
    pat_neg = log_pattern[:mid][::-1, :]

    # Общий θ = [0, ..., 90°] (одинаковый для обеих φ-половин)
    # Дополняем theta_neg нулём в начале, чтобы размеры совпали
    theta_neg = np.concatenate([[0.0], theta_neg])
    pat_neg = np.concatenate([log_pattern[mid : mid + 1, :], pat_neg], axis=0)

    phi_full = np.concatenate([phi, phi + np.pi])
    pat_full = np.concatenate([pat_pos, pat_neg], axis=1)

    TH, PH = np.meshgrid(theta_pos, phi_full, indexing="ij")
    U = np.sin(TH) * np.cos(PH)
    V = np.sin(TH) * np.sin(PH)

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw={"aspect": "equal"})
    ax.tripcolor(
        U.ravel(),
        V.ravel(),
        pat_full.ravel(),
        cmap="jet",
        vmin=-40,
        vmax=0,
        shading="gouraud",
    )
    sm = plt.cm.ScalarMappable(cmap="jet", norm=plt.Normalize(-40, 0))
    fig.colorbar(sm, ax=ax, label=r"$|F(\theta, \varphi)|$, дБ", shrink=0.8)

    for deg in [30, 60, 90]:
        r = np.sin(np.radians(deg))
        circle = plt.Circle(
            (0, 0), r, fill=False, color="gray", linewidth=0.5, linestyle="--"
        )
        ax.add_patch(circle)
        ax.text(
            0, r + 0.02, f"{deg}°", ha="center", va="bottom", fontsize=8, color="gray"
        )

    ax.set_xlabel(r"$u = \sin\theta\,\cos\varphi$")
    ax.set_ylabel(r"$v = \sin\theta\,\sin\varphi$")
    ax.set_title("ДН 2D антенной решётки")
    plt.tight_layout()
    plt.show()
