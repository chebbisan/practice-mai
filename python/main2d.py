import ctypes as ct
import logging
import platform
from pathlib import Path

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

SPEED_OF_LIGHT = 3e8

ROOT = Path(__file__).parent.parent
_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"


def main():
    logger.info("Starting 2D antenna array calculation")

    c_lib = initialize_library(str(LIB_PATH))

    Nx, Ny = 16, 16
    freq_0 = 3e9
    wave_length = SPEED_OF_LIGHT / freq_0
    wave_num = 2 * np.pi / wave_length
    logger.debug("Nx=%d, Ny=%d, freq_0=%.2e Hz, wave_length=%.4f m, wave_num=%.4f rad/m",
                 Nx, Ny, freq_0, wave_length, wave_num)

    theta_x = np.linspace(-np.pi / 2, np.pi / 2, 181)
    theta_y = np.linspace(-np.pi / 2, np.pi / 2, 181)
    logger.debug("theta_x: %d points, theta_y: %d points", theta_x.size, theta_y.size)

    delta_x = calculate_delta_x(wave_length, np.pi / 6)
    delta_y = calculate_delta_x(wave_length, np.pi / 6)
    x_arr = np.array([i * delta_x - delta_x * (Nx - 1) / 2 for i in range(Nx)])
    y_arr = np.array([i * delta_y - delta_y * (Ny - 1) / 2 for i in range(Ny)])
    logger.debug("delta_x=%.4f m, delta_y=%.4f m", delta_x, delta_y)

    f_arr = [complex_t(1, 0)] * theta_x.size
    c_f = list_to_c_complex_array(f_arr)
    c_x = list_to_c_double_array(x_arr)
    c_theta_x = list_to_c_double_array(theta_x)

    logger.info("Calling Calculate1DAntennaArray for row pattern (N=%d, points=%d)", Nx, theta_x.size)
    f_row = c_lib.Calculate1DAntennaArray(
        ct.c_int(Nx),
        ct.c_int(theta_x.size),
        c_f,
        c_x,
        c_theta_x,
        ct.c_double(wave_num),
    )
    logger.info("Row pattern calculated")

    c_y = list_to_c_double_array(y_arr)
    c_theta_y = list_to_c_double_array(theta_y)

    logger.info("Calling Calculate2DAntennaArray (Ny=%d, %dx%d grid)", Ny, theta_x.size, theta_y.size)
    result = c_lib.Calculate2DAntennaArray(
        ct.c_int(Ny),
        ct.c_int(theta_x.size),
        ct.c_int(theta_y.size),
        f_row,
        c_y,
        c_theta_x,
        c_theta_y,
        ct.c_double(wave_num),
    )
    logger.info("2D calculation complete")

    logger.debug("Building magnitude matrix (%dx%d)", theta_x.size, theta_y.size)
    pattern_2d = np.array(
        [
            abs(result[i * theta_y.size + j].real + 1j * result[i * theta_y.size + j].imag)
            for i in range(theta_x.size)
            for j in range(theta_y.size)
        ]
    ).reshape(theta_x.size, theta_y.size)

    log_pattern = np.clip(20 * np.log10(np.maximum(pattern_2d, 1e-10)), -40, 0)
    logger.debug("Peak pattern value: %.4f dB", log_pattern.max())

    TX, TY = np.meshgrid(np.degrees(theta_x), np.degrees(theta_y), indexing="ij")

    # --- Heatmap ---
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

    # --- 3D surface ---
    logger.info("Plotting 3D surface")
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

    # --- 3D balloon (linear scale) ---
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

    logger.debug("Freeing C memory")
    c_lib.FreeComplexArr(f_row)
    c_lib.FreeComplexArr(result)
    logger.info("Done")


if __name__ == "__main__":
    main()
