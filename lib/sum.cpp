#include "sum.hpp"

// Сумма двух действительных чисел
double Sum(double a, double b) {
    return a + b;
}

// Сумма двух комплексных чисел
complex_t SumComplex(complex_t a, complex_t b) {
    return complex_t{a.real + b.real, a.imag + b.imag};
}

// Сумма всех чисел в массиве. 
// Результат - массив с элементами равными сумме
double* SumArray(double* arr, int count) {
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

// Сумма всех чисел в массиве. 
// Результат - массив с элементами равными сумме
complex_t* SumComplexArray(complex_t* arr, int count) {
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

// Функция освобождения памяти
void FreeDoubleArr(double* arr) {
    delete[] arr;
}

// Функция освобождения памяти
void FreeComplexArr(complex_t* arr) {
    delete[] arr;
}
