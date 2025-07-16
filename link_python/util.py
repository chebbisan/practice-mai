import typing
from typing import List
from ctypes import c_int, c_double
from complex import complex_t

# Конвертация списка в C-массив
def ListToCIntegerArray(py_list: List):
    return (c_int * len(py_list))(*py_list)

# Конвертация списка в C-массив
def ListToCDoubleArray(py_list: List):
    return (c_double * len(py_list))(*py_list)

# Конвертация списка в C-массив
def ListToCComplexArray(py_list: List):
    return (complex_t * len(py_list))(*py_list)
