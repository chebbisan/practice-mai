import ctypes as ct
import logging
from typing import List

import numpy as np
from numpy import sin

try:
    from .complex import complex_t  # imported as package (e.g. tests)
except ImportError:
    from complex import complex_t  # run as script from python/

logger = logging.getLogger(__name__)


# Инициализация аргументов C-функций
def _initialize_arg_types(library):
    logger.debug("Initializing argtypes")
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
    library.Calculate1DAntennaArray.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.POINTER(complex_t),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
    ]
    library.Calculate2DAntennaArray.argtypes = [
        ct.c_int,
        ct.c_int,
        ct.c_int,
        ct.POINTER(complex_t),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_double,
    ]


# Инициализация возвращаемых типов C-функций
def _initialize_res_types(library):
    logger.debug("Initializing restypes")
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
    library.Calculate2DAntennaArray.restype = ct.POINTER(complex_t)


# Инициализация C-библиотеки
def initialize_library(path_to_lib):
    logger.info("Loading shared library: %s", path_to_lib)
    try:
        c_lib = ct.CDLL(path_to_lib)
    except OSError as e:
        logger.error("Failed to load library '%s': %s", path_to_lib, e)
        raise
    _initialize_arg_types(c_lib)
    _initialize_res_types(c_lib)
    logger.info("Library loaded successfully")
    return c_lib


# Конвертация списка в C-массив
def list_to_c_int_array(py_list: List):
    logger.debug("Converting list of %d ints to C array", len(py_list))
    return (ct.c_int * len(py_list))(*py_list)


# Конвертация списка в C-массив
def list_to_c_double_array(py_list: List):
    logger.debug("Converting list of %d doubles to C array", len(py_list))
    return (ct.c_double * len(py_list))(*py_list)


# Конвертация списка в C-массив
def list_to_c_complex_array(py_list: List):
    logger.debug("Converting list of %d complex_t to C array", len(py_list))
    return (complex_t * len(py_list))(*py_list)


def calculate_delta_x(wave_length, theta):
    return wave_length / (1 + sin(theta))


def calculate_1d_antenna_array(N, f_arr, x_arr, theta_arr, wave_num):
    logger.debug("Python 1D calculation: N=%d, theta_points=%d", N, theta_arr.size)
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
    logger.debug("Python 1D calculation complete")
    return radiation_pattern
