import ctypes as ct
from pathlib import Path
from complex import complex_t
from util import *

# Инициализация аргументов C-функций
def InitializeArgTypes(library):
    library.Sum.argtypes = [ct.c_double, ct.c_double]
    library.SumComplex.argtypes = [complex_t, complex_t]
    library.SumArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int]
    library.SumComplexArray.argtypes = [ct.POINTER(complex_t), ct.c_int]
    library.FreeDoubleArr.argtypes = [ct.POINTER(ct.c_double)]
    library.FreeComplexArr.argtypes = [ct.POINTER(complex_t)]

# Инициализация возвращаемых типов C-функций
def InitializeResTypes(library):
    library.Sum.restype = ct.c_double
    library.SumComplex.restype = complex_t
    library.SumArray.restype = ct.POINTER(ct.c_double)
    library.SumComplexArray.restype = ct.POINTER(complex_t)
    library.FreeDoubleArr.restype = None
    library.FreeComplexArr.restype = None

# Инициализация C-библиотеки
def InitializeLibrary(path_to_lib):
    c_lib = ct.CDLL(path_to_lib)
    InitializeArgTypes(c_lib)
    InitializeResTypes(c_lib)
    return c_lib


def main():
    # Путь до динамической библиотеки
    lib_name = "../build/libSum.so"
    c_lib = InitializeLibrary(lib_name)

    # Создание списка значений
    arr = [complex_t(1.0, 2.0), complex_t(3.0, 4.0), complex_t(5.0, 5.0)]
    # Расчет результата с использованием динамической библиотеки
    result = c_lib.SumComplexArray(ListToCComplexArray(arr), len(arr))

    # Вывод результата
    for i in range(len(arr)):
        print(result[i].real, result[i].imag)

    # Освобождение памяти
    c_lib.FreeComplexArr(result)


if __name__ == '__main__':
    main()
