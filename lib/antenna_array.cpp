#include "antenna_array.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>

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
            std::complex<double> f_elem(f_arr[j].real, f_arr[j].imag);
            auto exp_arg = -imag_unit * wave_num * x_arr[j] * std::sin(theta_arr[i]);
            part_sum += f_elem * std::exp(exp_arg);
        }
        radiation_pattern[i].real = part_sum.real() / N;
        radiation_pattern[i].imag = part_sum.imag() / N;
    }
    return radiation_pattern;
}

// Расчет двумерной антенной решетки с произвольным расположением элементов
// AF(θ_i, φ_j) = (1/N) * Σ f_n * exp(-jk(x_n·sinθ·cosφ + y_n·sinθ·sinφ))
complex_t* Calculate2DAntennaArray(int N, int n_theta, int n_phi, complex_t* f_arr, double* x_arr,
                                   double* y_arr, double* theta_arr, double* phi_arr,
                                   double wave_num) {
    static std::complex<double> imag_unit(0.0, 1.0);
    int total = n_theta * n_phi;
    complex_t* radiation_pattern = new complex_t[total];
    for (int i = 0; i < n_theta; ++i) {
        double sin_theta = std::sin(theta_arr[i]);
        for (int j = 0; j < n_phi; ++j) {
            double cos_phi = std::cos(phi_arr[j]);
            double sin_phi = std::sin(phi_arr[j]);
            std::complex<double> part_sum(0.0, 0.0);
            for (int n = 0; n < N; ++n) {
                std::complex<double> f_elem(f_arr[n].real, f_arr[n].imag);
                double phase =
                    wave_num * (x_arr[n] * sin_theta * cos_phi + y_arr[n] * sin_theta * sin_phi);
                part_sum += f_elem * std::exp(-imag_unit * phase);
            }
            int idx = i * n_phi + j;
            radiation_pattern[idx].real = part_sum.real() / N;
            radiation_pattern[idx].imag = part_sum.imag() / N;
        }
    }
    return radiation_pattern;
}

// Извлечение модуля из массива комплексных чисел
void ComplexArrayMagnitude(complex_t* arr, double* out, int n) {
    for (int i = 0; i < n; ++i) {
        out[i] = std::abs(std::complex<double>(arr[i].real, arr[i].imag));
    }
}

// Нормировка массива in-place, возвращает max
double NormalizeArray(double* arr, int n) {
    double max_val = arr[0];
    for (int i = 1; i < n; ++i) {
        if (arr[i] > max_val) max_val = arr[i];
    }
    if (max_val > 0.0) {
        for (int i = 0; i < n; ++i) {
            arr[i] /= max_val;
        }
    }
    return max_val;
}

// КНД 1D (формула 10.41): D₀ = 2 / ∫ F²(θ)cos(θ) dθ (трапеции)
double CalculateDirectivity1D(double* pattern, double* theta, int n_theta) {
    double integral = 0.0;
    for (int i = 0; i < n_theta - 1; ++i) {
        double dt = theta[i + 1] - theta[i];
        double f0 = pattern[i] * pattern[i] * std::cos(theta[i]);
        double f1 = pattern[i + 1] * pattern[i + 1] * std::cos(theta[i + 1]);
        integral += (f0 + f1) * 0.5 * dt;
    }
    return 2.0 / integral;
}

// КНД 2D (формула 10.40): D₀ = 4π / ∫∫ F²(θ,φ)cos(θ) dθ dφ
// pattern — row-major [n_theta × n_phi]
double CalculateDirectivity2D(double* pattern, double* theta, double* phi, int n_theta, int n_phi) {
    // Интегрируем по θ для каждого φ_j
    double* inner = new double[n_phi];
    for (int j = 0; j < n_phi; ++j) {
        double sum = 0.0;
        for (int i = 0; i < n_theta - 1; ++i) {
            double dt = theta[i + 1] - theta[i];
            double f0 = pattern[i * n_phi + j] * pattern[i * n_phi + j] * std::cos(theta[i]);
            double f1 = pattern[(i + 1) * n_phi + j] * pattern[(i + 1) * n_phi + j] *
                        std::cos(theta[i + 1]);
            sum += (f0 + f1) * 0.5 * dt;
        }
        inner[j] = sum;
    }
    // Интегрируем по φ
    double integral = 0.0;
    for (int j = 0; j < n_phi - 1; ++j) {
        double dp = phi[j + 1] - phi[j];
        integral += (inner[j] + inner[j + 1]) * 0.5 * dp;
    }
    delete[] inner;
    return 4.0 * M_PI / integral;
}
