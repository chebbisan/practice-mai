#pragma once

#include "complex.hpp"

extern "C" {
    
double Sum(double a, double b);
complex_t SumComplex(complex_t a, complex_t b);

double* SumArray(double* arr, int count);
complex_t* SumComplexArray(complex_t* arr, int count);

void FreeDoubleArr(double* arr);
void FreeComplexArr(complex_t* arr);

};