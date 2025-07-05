#include <iostream>
#include "../lib/sum.hpp"

/*
1. Бибилиотека на c++ (dll)
    Sum(double, double)
    SumComplex()
2. SumComplexArray() -> sum_array
3. Sum(func* )

4. Диаграмма решетки
F(theta, phi) = K * Sum(A_n * exp(-j ))
+ еще тестовая версия в MathCad
Сроки: до след чт, первые 3 пункта. 

GitHub: Virtual Machine работать тут
*/

int main() {
    complex_t* c_arr = new complex_t[10];
    for (int i = 0; i < 10; ++i) {
        c_arr[i].real = i;
        c_arr[i].imag = -i;
    }
    auto* sum_arr = SumComplexArray(c_arr, 10);
    for (int i = 0; i < 10; ++i) {
        std::cout << sum_arr[i].real << ' ' << sum_arr[i].imag << std::endl;
    }

    delete[] c_arr;
    delete[] sum_arr;
    return 0;
}