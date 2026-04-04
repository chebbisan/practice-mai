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

// Расчет двумерной антенной решетки с произвольным расположением элементов
// Результат — плоский массив размера n_theta * n_phi (row-major, theta — внешний индекс)
complex_t* Calculate2DAntennaArray(int N, int n_theta, int n_phi, complex_t* f_arr, double* x_arr,
                                   double* y_arr, double* theta_arr, double* phi_arr,
                                   double wave_num);

// Извлечение модуля из массива комплексных чисел: out[i] = |arr[i]|
void ComplexArrayMagnitude(complex_t* arr, double* out, int n);

// Нормировка массива in-place: arr[i] /= max. Возвращает max.
double NormalizeArray(double* arr, int n);

// КНД 1D (формула 10.41): D₀ = 2 / ∫ F²(θ)cos(θ) dθ
double CalculateDirectivity1D(double* pattern, double* theta, int n_theta);

// КНД 2D (формула 10.40): D₀ = 4π / ∫∫ F²(θ,φ)cos(θ) dθ dφ
// pattern — row-major [n_theta × n_phi]
double CalculateDirectivity2D(double* pattern, double* theta, double* phi, int n_theta, int n_phi);

// Освобождение double-массива
void FreeDoubleArr(double* arr);
};
