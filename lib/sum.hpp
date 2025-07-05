#pragma once

#include "complex.hpp"

extern "C" double Sum(double a, double b);
extern "C" complex_t SumComplex(complex_t a, complex_t b);

extern "C" double* SumArray(double* arr, int count);
extern "C" complex_t* SumComplexArray(complex_t* arr, int count);

extern "C" void FreeDoubleArr(double* arr);
extern "C" void FreeComplexArr(complex_t* arr);