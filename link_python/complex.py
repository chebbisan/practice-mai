import ctypes as ct

# Структура комплексного числа
class complex_t(ct.Structure):
    _fields_ = [("real", ct.c_double), ("imag", ct.c_double)]
