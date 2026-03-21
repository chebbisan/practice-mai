#include "../lib/sum.hpp"

#include <iostream>


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