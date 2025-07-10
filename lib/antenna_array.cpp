#include "antenna_array.hpp"

double CalculateNormalizingCoeff(complex_t* arr, int count) {
    double sum_abs = 0.0;
    for (int i = 0; i < count; ++i) {
        sum_abs += ComplexAbs(arr[i]);
    }
    return 1. / sum_abs;
}

double CalculateWaveNumber(double num, int phys) {
    if (phys == 1) {
        return 2 * M_PI * num / LIGHT_SPEED;
    } else if (phys == 2) {
        return 2 * M_PI / num;
    }
    throw std::runtime_error("Wrong argument");
}

uint64_t CalculateAntennaArraySize(uint64_t Nx, uint64_t Ny) {
    return Nx * Ny;
}

double* GenerateThetaRange(double dTheta = 1.0) {
    int count = 180 / dTheta;
    auto* theta_range = new double[count];
    double start_val = 0.0;
    for (int i = 0; i < count; ++i) {
        theta_range[i] = start_val;
        start_val += dTheta;
    }
    return theta_range;
}

double* GeneratePhiRange(double dPhi = 1.0) {
    int count = 360 / dPhi;
    auto* phi_range = new double[count];
    double start_val = 0.0;
    for (int i = 0; i < count; ++i) {
        phi_range[i] = start_val;
        start_val += dPhi;
    }
    return phi_range;
}

