#include <pybind11/pybind11.h>

int add(int i, int j) {
    return i + j;
}

/**
 * pip install pybind11
 *
 * g++ -O3 -Wall -shared -std=c++11 -fPIC $(python3 -m pybind11 --includes) example.cpp -o example$(python3-config --extension-suffix)
 * Building the above C++ code will produce a binary module file that can be imported to Python.
 */

PYBIND11_MODULE(example, m) {
    m.doc() = "pybind11 example plugin"; // optional module docstring

    m.def("add", &add, "A function that adds two numbers");
}