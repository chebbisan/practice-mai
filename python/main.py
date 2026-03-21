import ctypes as ct
import logging
import platform
from pathlib import Path

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
_LIB_SUFFIX = ".dylib" if platform.system() == "Darwin" else ".so"
LIB_PATH = ROOT / "build" / f"libAntennaArray{_LIB_SUFFIX}"
CONFIG_PATH = Path(__file__).parent / "config.yaml"


def main():
    logger.info("Starting 1D antenna array calculation")

    with open(CONFIG_PATH) as f:
        cfg = yaml.safe_load(f)["array_1d"]
    N         = cfg["N"]
    freq_0    = cfg["freq_hz"]
    d         = cfg["d"]
    steer_deg = cfg["steer_deg"]
    n_theta   = cfg["n_theta"]

    c_lib = initialize_library(str(LIB_PATH))

    wave_length = SPEED_OF_LIGHT / freq_0
    wave_num = 2 * np.pi / wave_length
    logger.debug("N=%d, freq_0=%.2e Hz, wave_length=%.4f m, wave_num=%.4f rad/m",
                 N, freq_0, wave_length, wave_num)

    theta_range = np.linspace(-np.pi / 2, np.pi / 2, n_theta)
    f_arr = [complex_t(1, 0) for _ in range(N)]
    delta_x = d if d is not None else calculate_delta_x(wave_length, np.radians(steer_deg))
    L = delta_x * (N - 1)
    x_arr = np.array([(i * delta_x) - L / 2 for i in range(N)])
    logger.debug("delta_x=%.4f m, array aperture L=%.4f m", delta_x, L)

    c_theta_range = list_to_c_double_array(theta_range)
    c_f_arr = list_to_c_complex_array(f_arr)
    c_x_arr = list_to_c_double_array(x_arr)

    logger.info("Calling Calculate1DAntennaArray (N=%d, theta_points=%d)", N, theta_range.size)
    radiation_pattern = c_lib.Calculate1DAntennaArray(
        ct.c_int(N),
        ct.c_int(theta_range.size),
        c_f_arr,
        c_x_arr,
        c_theta_range,
        ct.c_double(wave_num),
    )
    logger.info("C library calculation complete")

    abs_rad_pattern = np.array([
        np.abs(complex(radiation_pattern[i].real, radiation_pattern[i].imag))
        for i in range(theta_range.size)
    ])

    log_abs_rad_pattern = 20 * np.log10(abs_rad_pattern)
    log_ray_width = -3
    log_side_ray = -13
    logger.debug("Peak pattern value: %.4f dB", log_abs_rad_pattern.max())

    logger.info("Plotting radiation pattern")
    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(theta_range), log_abs_rad_pattern)
    plt.axhline(log_ray_width, color="red", label=f"{log_ray_width} дБ", linestyle="--")
    plt.axhline(log_side_ray, color="green", label=f"{log_side_ray} дБ", linestyle="--")
    plt.xlabel(r"$\theta$, градус")
    plt.ylabel(r"$|F(\theta)|$, дБ", rotation=0)
    plt.legend()
    plt.show()

    logger.debug("Freeing C memory")
    c_lib.FreeComplexArr(radiation_pattern)
    logger.info("Done")


if __name__ == "__main__":
    main()
