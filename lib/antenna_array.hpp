#pragma once

#include "complex.hpp"

#define FREQUENCY 1
#define WAVELENGTH 2
#define LIGHT_SPEED 300000000

extern "C" {

// Расчет нормализующего коэффициента
double CalculateNormalizingCoeff(complex_t* arr, int count);

// Расчет волнового числа
double CalculateWaveNumber(double num, int phys);

// Перевод углов в радианы
double DegreesToRadians(double degrees);

// Перевод радиан в углы
double RadiansToDegrees(double radians);

// Расчет одномерной антенной решетки
complex_t* Calculate1DAntennaArray(int N, int size, complex_t* f_arr, double* x_arr,
                                   double* theta_arr, double wave_num);
};
