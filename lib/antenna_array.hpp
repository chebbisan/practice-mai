#pragma once

#include "complex.hpp"

#include <cstdint>

#define FREQUENCY 1
#define WAVELENGTH 2
#define LIGHT_SPEED 300000000

extern "C" {

// Расчет нормализующего коэффициента
double CalculateNormalizingCoeff(complex_t* arr, int count);

// Расчет волнового числа
double CalculateWaveNumber(double num, int phys);

// Расчет количества ячеек в решетке
uint64_t CalculateAntennaArraySize(uint64_t Nx, uint64_t Ny);

// Расчет шага по оси X
double CalculateDeltaX(double wave_length, double theta_x);

// Расчет шага по оси Y
double CalculateDeltaY(double wave_length, double theta_y);

// Перевод углов в радианы
double DegreesToRadians(double degrees);

// Перевод радиан в углы
double RadiansToDegrees(double radians);

// Расчет одномерной антенной решетки
complex_t* Calculate1DAntennaArray(int N, int size, complex_t* f_arr, double* x_arr,
                                   double* theta_arr, double wave_num);

// Расчет двумерной антенной решетки
// Результат — плоский массив [size_x * size_y] в порядке [i_theta_x][i_theta_y]
complex_t* Calculate2DAntennaArray(int Ny, int size_x, int size_y, complex_t* f_row,
                                   double* y_arr, double* theta_x_arr,
                                   double* theta_y_arr, double wave_num);
};
