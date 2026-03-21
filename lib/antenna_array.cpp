#include "antenna_array.hpp"

#include <stdexcept>
#include <vector>

// Расчет нормализующего коэффициента
double CalculateNormalizingCoeff(complex_t* arr, int count) {
    double sum_abs = 0.0;
    for (int i = 0; i < count; ++i) {
        sum_abs += std::abs(std::complex<double>(arr[i].real, arr[i].imag));
    }
    return 1. / sum_abs;
}

// Расчет волнового числа
double CalculateWaveNumber(double num, int phys) {
    if (phys == FREQUENCY) {
        return 2 * M_PI * num / LIGHT_SPEED;
    } else if (phys == WAVELENGTH) {
        return 2 * M_PI / num;
    }
    throw std::runtime_error("Wrong argument");
}

// Расчет количества ячеек в решетке
uint64_t CalculateAntennaArraySize(uint64_t Nx, uint64_t Ny) { return Nx * Ny; }

// Расчет шага по оси X
double CalculateDeltaX(double wave_length, double theta_x) {
    return wave_length / (1. + sin(theta_x));
}

// Расчет шага по оси Y
double CalculateDeltaY(double wave_length, double theta_y) {
    return wave_length / (1. + sin(theta_y));
}

// Перевод углов в радианы
double DegreesToRadians(double degrees) { return degrees * (M_PI / 180); }

// Перевод радиан в углы
double RadiansToDegrees(double radians) { return radians * (180 / M_PI); }

// Расчет одномерной антенной решетки
complex_t* Calculate1DAntennaArray(int N, int size, complex_t* f_arr, double* x_arr,
                                   double* theta_arr, double wave_num) {
    static std::complex<double> imag_unit(0.0, 1.0);
    complex_t* radiation_pattern = new complex_t[size];
    for (int i = 0; i < size; ++i) {
        std::complex<double> part_sum(0.0, 0.0);
        for (int j = 0; j < N; ++j) {
            std::complex<double> f_elem(f_arr[i].real, f_arr[i].imag);
            auto exp_arg = -imag_unit * wave_num * x_arr[j] * std::sin(theta_arr[i]);
            part_sum += f_elem * std::exp(exp_arg);
        }
        radiation_pattern[i].real = part_sum.real() / N;
        radiation_pattern[i].imag = part_sum.imag() / N;
    }
    return radiation_pattern;
}

// Расчет двумерной антенной решетки
// Диаграмма строки (f_row) используется как элементная диаграмма для столбца.
// F_2D(θ_x, θ_y) = F_row(θ_x) · (1/Ny) · Σ_m exp(−j·k·y_m·sin(θ_y))
complex_t* Calculate2DAntennaArray(int Ny, int size_x, int size_y, complex_t* f_row,
                                   double* y_arr, double* theta_x_arr,
                                   double* theta_y_arr, double wave_num) {
    static std::complex<double> imag_unit(0.0, 1.0);

    // Предварительный расчет множителей столбца для каждого θ_y
    std::vector<std::complex<double>> col_factor(size_y);
    for (int j = 0; j < size_y; ++j) {
        std::complex<double> sum(0.0, 0.0);
        for (int m = 0; m < Ny; ++m) {
            sum += std::exp(-imag_unit * wave_num * y_arr[m] * std::sin(theta_y_arr[j]));
        }
        col_factor[j] = sum / static_cast<double>(Ny);
    }

    // F_2D[i * size_y + j] = F_row[i] * col_factor[j]
    complex_t* result = new complex_t[size_x * size_y];
    for (int i = 0; i < size_x; ++i) {
        std::complex<double> f_r(f_row[i].real, f_row[i].imag);
        for (int j = 0; j < size_y; ++j) {
            std::complex<double> val = f_r * col_factor[j];
            result[i * size_y + j].real = val.real();
            result[i * size_y + j].imag = val.imag();
        }
    }
    return result;
}
