#pragma once

#include "complex.hpp"
#include <cstdint>

#define FREQUENCY 1
#define WAVELENGTH 2
#define LIGHT_SPEED 300000000



extern "C" {

double CalculateNormalizingCoeff(double* arr, int count);
double CalculateWaveNumber(double num, int phys);
uint64_t CalculateAntennaArraySize(uint64_t Nx, uint64_t Ny);
double* GenerateThetaRange(double dTheta);
double* GeneratePhiRange(double dPhi);
/*
    [r_x * cos(phi) + r_y * sin(phi)] * x * sin(theta) + r_z * cos(theta)
*/

};
