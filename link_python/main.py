import ctypes as ct
from pathlib import Path

class complex_t(ct.Structure):
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]

def InitializeArgTypes(library):
    library.Sum.argtypes = [ct.c_double, ct.c_double]
    library.SumComplex.argtypes = [complex_t, complex_t]
    library.SumArray.argtypes = [ct.POINTER(ct.c_double), ct.c_int]
    library.SumComplexArray.argtypes = [ct.POINTER(complex_t), ct.c_int]
    library.FreeDoubleArr.argtypes = [ct.POINTER(ct.c_double)]
    library.FreeComplexArr.argtypes = [ct.POINTER(complex_t)]

def InitializeResTypes(library):
    library.Sum.restype = ct.c_double
    library.SumComplex.restype = complex_t
    library.SumArray.restype = ct.POINTER(ct.c_double)
    library.SumComplexArray.restype = ct.POINTER(complex_t)
    library.FreeDoubleArr.restype = None
    library.FreeComplexArr.restype = None

def InitializeLibrary(path_to_lib):
    c_lib = ct.CDLL(path_to_lib)
    InitializeArgTypes(c_lib)
    InitializeResTypes(c_lib)
    return c_lib

def main():
    lib_name = "../build/libSum.so"
    c_lib = InitializeLibrary(lib_name)
    size = 3
    arr = (complex_t * size)(complex_t(1.0, 2.0), complex_t(3.0, 4.0), complex_t(5.0, 5.0))
    result = c_lib.SumComplexArray(arr, size)
    for i in range(size):
        print(result[i].real, result[i].imag)

    c_lib.FreeComplexArr(result)

if __name__ == '__main__':
    main()