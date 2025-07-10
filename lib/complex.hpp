#pragma once

#include <iostream>
#include <complex>
#include <cmath>

struct complex_t {
    double real, imag;
};

extern "C" {

double ComplexAbs(complex_t c);

};