from pathlib import Path
from complex import complex_t
from util import *
import numpy as np
import matplotlib.pyplot as plt

FREQUENCY = 1 # for calculating wavenum
WAVELENGTH = 2 # for calculating wavenum

def main():
    # Путь до динамической библиотеки
    lib_name = "../build/libAntennaArray.so"
    c_lib = InitializeLibrary(lib_name)


    N = 100
    freq_0 = 3 * 10**9
    wave_length = (3 * 10**8) / freq_0
    wave_num = c_lib.CalculateWaveNumber(ct.c_double(freq_0), ct.c_int(FREQUENCY))

    theta_range = np.linspace(0, np.pi, 100)
    f_arr = [complex_t(np.cos(theta), 0) for theta in theta_range]
    c_theta_range = ListToCDoubleArray(list(theta_range))
    c_f_arr = ListToCComplexArray(f_arr)

    delta_x = c_lib.CalculateDeltaX(ct.c_double(wave_length), ct.c_double(30))
    x_arr = [i * delta_x for i in range(N)]
    c_x_arr = ListToCDoubleArray(x_arr)

    # print(theta_range)

    abs_rad_pattern = np.zeros_like(theta_range)
    radiation_pattern = c_lib.Calculate1DAntennaArray(ct.c_int(N), ct.c_int(theta_range.size), c_f_arr, c_x_arr, c_theta_range, ct.c_double(wave_num))
    for i in range(N):
        # print(radiation_pattern[i].real, radiation_pattern[i].imag, np.abs(radiation_pattern[i].real + 1j * radiation_pattern[i].imag))
        abs_rad_pattern[i] = np.abs(radiation_pattern[i].real + 1j * radiation_pattern[i].imag)

    plt.figure(figsize=(8,6))
    plt.plot(theta_range, 20 * np.log(abs_rad_pattern))
    plt.show()
    

    c_lib.FreeComplexArr(radiation_pattern)


    # # Создание списка значений
    # arr = [complex_t(1.0, 2.0), complex_t(3.0, 4.0), complex_t(5.0, 5.0)]
    # # Расчет результата с использованием динамической библиотеки
    # result = c_lib.SumComplexArray(ListToCComplexArray(arr), len(arr))

    # # Вывод результата
    # for i in range(len(arr)):
    #     print(result[i].real, result[i].imag)

    # # Освобождение памяти
    # c_lib.FreeComplexArr(result)


if __name__ == '__main__':
    main()
