#pragma once

#include "complex.hpp"

extern "C" {

// Сумма двух действительных чисел
double Sum(double a, double b);

// Сумма двух комплексных чисел
complex_t SumComplex(complex_t a, complex_t b);

// Сумма всех чисел в массиве. 
// Результат - массив с элементами равными сумме
double* SumArray(double* arr, int count);

// Сумма всех чисел в массиве. 
// Результат - массив с элементами равными сумме
complex_t* SumComplexArray(complex_t* arr, int count);

// Функция освобождения памяти
void FreeDoubleArr(double* arr);

// Функция освобождения памяти
void FreeComplexArr(complex_t* arr);

};