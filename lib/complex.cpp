#include "complex.hpp"

// Модуль комплексного числа
double ComplexAbs(complex_t c) {
    return sqrt(c.real*c.real + c.imag*c.imag);
}
