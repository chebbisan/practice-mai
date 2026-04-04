import logging

import matplotlib.pyplot as plt
import numpy as np

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logging.getLogger("matplotlib").setLevel(logging.WARNING)
logger = logging.getLogger(__name__)


def plot_1d_patterns(theta_x, theta_y, log_row, log_col):
    logger.info("Plotting 1D row and column patterns")
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    ax1.plot(np.degrees(theta_x), log_row)
    ax1.set_xlabel(r"$\theta_x$, градус")
    ax1.set_ylabel(r"$|F(\theta_x)|$, дБ", rotation=0)
    ax1.set_title("Диаграмма строки")
    ax2.plot(np.degrees(theta_y), log_col)
    ax2.set_xlabel(r"$\theta_y$, градус")
    ax2.set_ylabel(r"$|F(\theta_y)|$, дБ", rotation=0)
    ax2.set_title("Диаграмма столбца")
    plt.tight_layout()
    plt.show()


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


def plot_3d_surface(theta_x, theta_y, log_pattern):
    logger.info("Plotting 3D surface")
    TX, TY = np.meshgrid(np.degrees(theta_x), np.degrees(theta_y), indexing="ij")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(
        TY, TX, log_pattern, cmap="jet", vmin=-40, vmax=0, antialiased=True
    )
    fig.colorbar(
        surf, ax=ax, shrink=0.5, pad=0.1, label=r"$|F(\theta_x, \theta_y)|$, дБ"
    )
    ax.set_xlabel(r"$\theta_y$, градус")
    ax.set_ylabel(r"$\theta_x$, градус")
    ax.set_zlabel(r"$|F|$, дБ")
    ax.set_title("Диаграмма направленности 2D антенной решётки (3D)")
    plt.tight_layout()
    plt.show()


def plot_3d_balloon(theta_x, theta_y, pattern_2d):
    logger.info("Plotting 3D balloon pattern")
    lin_pattern = pattern_2d / pattern_2d.max()
    TX_rad, TY_rad = np.meshgrid(theta_x, theta_y, indexing="ij")
    X = lin_pattern * np.cos(TX_rad) * np.cos(TY_rad)
    Y = lin_pattern * np.cos(TX_rad) * np.sin(TY_rad)
    Z = lin_pattern * np.sin(TX_rad)

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(lin_pattern), antialiased=True)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("Диаграмма направленности 2D антенной решётки (шар)")
    fig.colorbar(
        plt.cm.ScalarMappable(cmap="jet"),
        ax=ax,
        shrink=0.5,
        pad=0.1,
        label=r"$|F|$ (норм.)",
    )
    plt.tight_layout()
    plt.show()
