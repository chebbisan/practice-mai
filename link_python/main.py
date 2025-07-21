from pathlib import Path
from complex import complex_t
from util import *
import numpy as np
import matplotlib.pyplot as plt

SPEED_OF_LIGHT = 3 * 10**8

def main():
    # Путь до динамической библиотеки
    lib_name = "../build/libAntennaArray.so"
    c_lib = InitializeLibrary(lib_name)

    # Подсчет необходимых значений
    N = 16
    freq_0 = 3 * 10**9
    wave_length = (SPEED_OF_LIGHT) / freq_0
    wave_num = 2 * np.pi / wave_length

    theta_range = np.linspace(0, 2*np.pi, 1001)
    f_arr = [complex_t(1, 0) for _ in theta_range]
    delta_x = CalculateDeltaX(wave_length, np.pi / 6)
    L = delta_x * (N - 1)
    x_arr = [(i * delta_x) - L/2 for i in range(N)]
    

    c_theta_range = ListToCDoubleArray(list(theta_range))
    c_f_arr = ListToCComplexArray(f_arr)
    c_x_arr = ListToCDoubleArray(x_arr)

    abs_rad_pattern = np.zeros_like(theta_range)
    radiation_pattern = c_lib.Calculate1DAntennaArray(ct.c_int(N), ct.c_int(
        theta_range.size), c_f_arr, c_x_arr, c_theta_range, ct.c_double(wave_num))
    for i in range(theta_range.size):
        abs_rad_pattern[i] = np.abs(
            radiation_pattern[i].real + 1j * radiation_pattern[i].imag)

    log_abs_rad_pattern = 20 * np.log10(abs_rad_pattern)
    log_ray_width = -3
    log_side_ray = -13

    rad_table = dict(zip(theta_range, abs_rad_pattern))
    

    # Построение графика
    plt.figure(figsize=(8, 6))
    plt.plot(np.degrees(theta_range) - 180, log_abs_rad_pattern)
    plt.axhline(log_ray_width, color='red', label=f"{log_ray_width} дБ", linestyle='--')
    plt.axhline(log_side_ray, color='green', label=f"{log_side_ray} дБ", linestyle='--')
    plt.xlabel("Degrees")
    plt.ylabel("F(theta)")
    plt.legend()
    plt.show()

    # Освобождение памяти
    c_lib.FreeComplexArr(radiation_pattern)


if __name__ == '__main__':
    main()
