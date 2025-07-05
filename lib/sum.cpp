#include "sum.hpp"

extern "C" double Sum(double a, double b) {
    return a + b;
}

extern "C" complex_t SumComplex(complex_t a, complex_t b) {
    return complex_t{a.real + b.real, a.imag + b.imag};
}

extern "C" double* SumArray(double* arr, int count) {
    double sum = 0;
    for (int i = 0; i < count; ++i) {
        sum += arr[i];
    }
    std::cerr << "Creating new array\n";
    auto* result_arr = new double[count];
    for (int i = 0; i < count; ++i) {
        result_arr[i] = sum;
    }
    return result_arr;
}

extern "C" complex_t* SumComplexArray(complex_t* arr, int count) {
    complex_t sum{0, 0};
    for (int i = 0; i < count; ++i) {
        sum.real += arr[i].real;
        sum.imag += arr[i].imag;
    }
    std::cerr << "Creating new array\n";
    auto* result_arr = new complex_t[count];
    for (int i = 0; i < count; ++i) {
        result_arr[i] = sum;
    }
    return result_arr;
}

extern "C" void FreeDoubleArr(double* arr) {
    delete[] arr;
}
extern "C" void FreeComplexArr(complex_t* arr) {
    delete[] arr;
}