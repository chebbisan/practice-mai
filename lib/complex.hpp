#pragma once

#include <iostream>
#include <complex>
#include <cmath>

// Структура для описания комплексного числа
struct complex_t {
    double real, imag;
};

extern "C" {

// Модуль комплексного числа
double ComplexAbs(complex_t c);

};