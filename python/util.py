import typing
from typing import List
import ctypes as ct
from .complex import complex_t
from numpy import sin
import numpy as np

# Инициализация аргументов C-функций
def InitializeArgTypes(library):
    library.Sum.argtypes = [ct.c_double, ct.c_double]
    library.SumComplex.argtypes = [complex_t, complex_t]
    library.SumArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int]
    library.SumComplexArray.argtypes = [ct.POINTER(complex_t), ct.c_int]
    library.FreeDoubleArr.argtypes = [ct.POINTER(ct.c_double)]
    library.FreeComplexArr.argtypes = [ct.POINTER(complex_t)]

    library.CalculateNormalizingCoeff.argtypes = [ct.POINTER(complex_t), ct.c_int]
    library.CalculateWaveNumber.argtypes = [ct.c_double, ct.c_int]
    library.CalculateAntennaArraySize.argtypes = [ct.c_uint64, ct.c_uint64]
    library.CalculateDeltaX.argtypes = [ct.c_double, ct.c_double]
    library.CalculateDeltaY.argtypes = [ct.c_double, ct.c_double]
    library.DegreesToRadians.argtypes = [ct.c_double]
    library.RadiansToDegrees.argtypes = [ct.c_double]
    library.Calculate1DAntennaArray.argtypes = [ct.c_int, ct.c_int, ct.POINTER(complex_t), ct.POINTER(ct.c_double), ct.POINTER(ct.c_double), ct.c_double]

# Инициализация возвращаемых типов C-функций
def InitializeResTypes(library):
    library.Sum.restype = ct.c_double
    library.SumComplex.restype = complex_t
    library.SumArray.restype = ct.POINTER(ct.c_double)
    library.SumComplexArray.restype = ct.POINTER(complex_t)
    library.FreeDoubleArr.restype = None
    library.FreeComplexArr.restype = None

    library.CalculateNormalizingCoeff.restype = ct.c_double
    library.CalculateWaveNumber.restype = ct.c_double
    library.CalculateAntennaArraySize.restype = ct.c_uint64
    library.CalculateDeltaX.restype = ct.c_double
    library.CalculateDeltaY.restype = ct.c_double
    library.DegreesToRadians.restype = ct.c_double
    library.RadiansToDegrees.restype = ct.c_double
    library.Calculate1DAntennaArray.restype = ct.POINTER(complex_t)

# Инициализация C-библиотеки
def InitializeLibrary(path_to_lib):
    c_lib = ct.CDLL(path_to_lib)
    InitializeArgTypes(c_lib)
    InitializeResTypes(c_lib)
    return c_lib


# Конвертация списка в C-массив
def ListToCIntegerArray(py_list: List):
    return (ct.c_int * len(py_list))(*py_list)

# Конвертация списка в C-массив
def ListToCDoubleArray(py_list: List):
    return (ct.c_double * len(py_list))(*py_list)

# Конвертация списка в C-массив
def ListToCComplexArray(py_list: List):
    return (complex_t * len(py_list))(*py_list)

def CalculateDeltaX(wave_length, theta):
    return wave_length / (1 + sin(theta))

def Calculate1DAntennaArray(N, f_arr, x_arr, theta_arr, wave_num):
    imag_unit = np.complex128(0 + 1j)
    radiation_pattern = [complex_t(1, 0)] * theta_arr.size
    for i in range(theta_arr.size):
        part_sum = np.complex128(0 + 1j * 0)
        for j in range(N):
            f_elem = np.complex128(f_arr[i].real + 1j * f_arr[j].imag)
            exp_arg = -imag_unit * wave_num * x_arr[j] * np.sin(theta_arr[i])
            part_sum += f_elem * np.exp(exp_arg)
        radiation_pattern[i].real = part_sum.real / N
        radiation_pattern[i].imag = part_sum.imag / N
    return radiation_pattern