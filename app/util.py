import ctypes as ct
import logging
import platform
from pathlib import Path
from typing import List

import numpy as np
from numpy import sin

try:
    from .complex import complex_t  # imported as package (e.g. tests)
except ImportError:
    from complex import complex_t  # run as script from app/

logger = logging.getLogger(__name__)

_LIB_SUFFIX = {"Darwin": ".dylib", "Windows": ".dll"}.get(platform.system(), ".so")
_LIB_PATH = Path(__file__).parent.parent / "build" / f"libAntennaArray{_LIB_SUFFIX}"
_LIB = None


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
        ct.c_int,  # N
        ct.c_int,  # n_theta
        ct.c_int,  # n_phi
        ct.POINTER(complex_t),  # f_arr
        ct.POINTER(ct.c_double),  # x_arr
        ct.POINTER(ct.c_double),  # y_arr
        ct.POINTER(ct.c_double),  # theta_arr
        ct.POINTER(ct.c_double),  # phi_arr
        ct.c_double,  # wave_num
    ]
    library.Calculate3DAntennaArray.argtypes = [
        ct.c_int,  # N
        ct.c_int,  # n_theta
        ct.c_int,  # n_phi
        ct.POINTER(complex_t),  # f_arr
        ct.POINTER(ct.c_double),  # x_arr
        ct.POINTER(ct.c_double),  # y_arr
        ct.POINTER(ct.c_double),  # z_arr
        ct.POINTER(ct.c_double),  # theta_arr
        ct.POINTER(ct.c_double),  # phi_arr
        ct.c_double,  # wave_num
    ]
    library.FullPipeline3D.argtypes = [
        ct.c_int,  # N
        ct.c_int,  # n_theta
        ct.c_int,  # n_phi
        ct.POINTER(ct.c_double),  # x_arr
        ct.POINTER(ct.c_double),  # y_arr
        ct.POINTER(ct.c_double),  # z_arr
        ct.POINTER(ct.c_double),  # amplitudes
        ct.POINTER(ct.c_double),  # theta_arr
        ct.POINTER(ct.c_double),  # phi_arr
        ct.c_double,  # wave_num
    ]
    library.ComplexArrayMagnitude.argtypes = [
        ct.POINTER(complex_t),
        ct.POINTER(ct.c_double),
        ct.c_int,
    ]
    library.NormalizeArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int]
    library.CalculateDirectivity1D.argtypes = [
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_int,
    ]
    library.CalculateDirectivity2D.argtypes = [
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.POINTER(ct.c_double),
        ct.c_int,
        ct.c_int,
    ]
    library.FullPipeline1D.argtypes = [
        ct.c_int,  # N
        ct.c_int,  # n_theta
        ct.POINTER(ct.c_double),  # x_arr
        ct.POINTER(ct.c_double),  # amplitudes
        ct.POINTER(ct.c_double),  # theta_arr
        ct.c_double,  # wave_num
    ]
    library.FullPipeline2D.argtypes = [
        ct.c_int,  # N
        ct.c_int,  # n_theta
        ct.c_int,  # n_phi
        ct.POINTER(ct.c_double),  # x_arr
        ct.POINTER(ct.c_double),  # y_arr
        ct.POINTER(ct.c_double),  # amplitudes
        ct.POINTER(ct.c_double),  # theta_arr
        ct.POINTER(ct.c_double),  # phi_arr
        ct.c_double,  # wave_num
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
    library.DegreesToRadians.restype = ct.c_double
    library.RadiansToDegrees.restype = ct.c_double
    library.Calculate1DAntennaArray.restype = ct.POINTER(complex_t)
    library.Calculate2DAntennaArray.restype = ct.POINTER(complex_t)
    library.Calculate3DAntennaArray.restype = ct.POINTER(complex_t)
    library.FullPipeline3D.restype = ct.c_double
    library.ComplexArrayMagnitude.restype = None
    library.NormalizeArray.restype = ct.c_double
    library.CalculateDirectivity1D.restype = ct.c_double
    library.CalculateDirectivity2D.restype = ct.c_double
    library.FullPipeline1D.restype = ct.c_double
    library.FullPipeline2D.restype = ct.c_double


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


def get_library():
    """Lazily load and cache the shared C++ library."""
    global _LIB
    if _LIB is None:
        if not _LIB_PATH.exists():
            raise FileNotFoundError(
                f"shared library not found: {_LIB_PATH} — build the project first"
            )
        _LIB = initialize_library(str(_LIB_PATH))
    return _LIB


def compute_2d_array_factor_cpp(
    x_arr: np.ndarray,
    y_arr: np.ndarray,
    amplitudes: np.ndarray,
    theta: np.ndarray,
    phi: np.ndarray,
    wave_num: float,
    z_arr: np.ndarray | None = None,
) -> np.ndarray:
    """Множитель решётки через C++ Calculate2D/3DAntennaArray. Возвращает (n_theta, n_phi) complex128, нормированный на N."""
    lib = get_library()
    N = int(x_arr.size)
    n_theta = int(theta.size)
    n_phi = int(phi.size)

    f_arr = (complex_t * N)()
    for i in range(N):
        f_arr[i].real = float(amplitudes[i])
        f_arr[i].imag = 0.0

    c_x = list_to_c_double_array(x_arr)
    c_y = list_to_c_double_array(y_arr)
    c_theta = list_to_c_double_array(theta)
    c_phi = list_to_c_double_array(phi)

    if z_arr is not None:
        c_z = list_to_c_double_array(z_arr)
        raw = lib.Calculate3DAntennaArray(
            ct.c_int(N),
            ct.c_int(n_theta),
            ct.c_int(n_phi),
            f_arr,
            c_x,
            c_y,
            c_z,
            c_theta,
            c_phi,
            ct.c_double(wave_num),
        )
    else:
        raw = lib.Calculate2DAntennaArray(
            ct.c_int(N),
            ct.c_int(n_theta),
            ct.c_int(n_phi),
            f_arr,
            c_x,
            c_y,
            c_theta,
            c_phi,
            ct.c_double(wave_num),
        )

    # complex_t == {double real, double imag} совпадает по layout с complex128
    total = n_theta * n_phi
    DoubleBuf = ct.c_double * (total * 2)
    raw_doubles = ct.cast(raw, ct.POINTER(DoubleBuf)).contents
    af = np.frombuffer(raw_doubles, dtype=np.complex128).reshape(n_theta, n_phi).copy()
    lib.FreeComplexArr(raw)
    return af


def calculate_delta_x(wave_length, theta):
    return wave_length / (1 + sin(theta))


def calculate_1d_antenna_array(N, f_arr, x_arr, theta_arr, wave_num):
    logger.debug("Python 1D calculation: N=%d, theta_points=%d", N, theta_arr.size)
    imag_unit = np.complex128(0 + 1j)
    radiation_pattern = [complex_t(0, 0) for _ in range(theta_arr.size)]
    for i in range(theta_arr.size):
        part_sum = np.complex128(0 + 1j * 0)
        for j in range(N):
            f_elem = np.complex128(f_arr[j].real + 1j * f_arr[j].imag)
            exp_arg = -imag_unit * wave_num * x_arr[j] * np.sin(theta_arr[i])
            part_sum += f_elem * np.exp(exp_arg)
        radiation_pattern[i].real = part_sum.real / N
        radiation_pattern[i].imag = part_sum.imag / N
    logger.debug("Python 1D calculation complete")
    return radiation_pattern
