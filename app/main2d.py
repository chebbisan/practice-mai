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
    ax1.axhline(-3, color="red", linestyle="--", label="-3 дБ")
    ax1.axhline(-13, color="green", linestyle="--", label="-13 дБ")
    ax1.set_xlabel(r"$\theta_x$, градус")
    ax1.set_ylabel(r"$|F(\theta_x)|$, дБ", rotation=0)
    ax1.set_title("Диаграмма строки")
    ax1.legend()
    ax2.plot(np.degrees(theta_y), log_col)
    ax2.axhline(-3, color="red", linestyle="--", label="-3 дБ")
    ax2.axhline(-13, color="green", linestyle="--", label="-13 дБ")
    ax2.set_xlabel(r"$\theta_y$, градус")
    ax2.set_ylabel(r"$|F(\theta_y)|$, дБ", rotation=0)
    ax2.set_title("Диаграмма столбца")
    ax2.legend()
    plt.tight_layout()
    plt.show()


def plot_heatmap(theta_x, theta_y, log_pattern):
    logger.info("Plotting heatmap")
    plt.figure(figsize=(8, 7))
    plt.imshow(
        log_pattern,
        extent=[
            np.degrees(theta_y[0]),
            np.degrees(theta_y[-1]),
            np.degrees(theta_x[-1]),
            np.degrees(theta_x[0]),
        ],
        aspect="auto",
        cmap="jet",
        vmin=-40,
        vmax=0,
    )
    plt.colorbar(label=r"$|F(\theta_x, \theta_y)|$, дБ")
    plt.xlabel(r"$\theta_y$, градус")
    plt.ylabel(r"$\theta_x$, градус")
    plt.title("Диаграмма направленности 2D антенной решётки (тепловая карта)")
    plt.tight_layout()
    plt.show()


def plot_3d_surface(theta_x, theta_y, log_pattern):
    logger.info("Plotting 3D surface")
    TX, TY = np.meshgrid(np.degrees(theta_x), np.degrees(theta_y), indexing="ij")
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")
    surf = ax.plot_surface(TY, TX, log_pattern, cmap="jet", vmin=-40, vmax=0, antialiased=True)
    fig.colorbar(surf, ax=ax, shrink=0.5, pad=0.1, label=r"$|F(\theta_x, \theta_y)|$, дБ")
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
    surf = ax.plot_surface(X, Y, Z, facecolors=plt.cm.jet(lin_pattern), antialiased=True)
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
