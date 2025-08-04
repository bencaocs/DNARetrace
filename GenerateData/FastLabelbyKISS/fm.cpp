#include <pybind11/pybind11.h>
#include <pybind11/stl.h>  // for STL container support
#include <pybind11/iostream.h>  // for input/output streams

#include "fm_index.hpp"  // Replace with the actual path to your FMIndex class

namespace py = pybind11;

PYBIND11_MODULE(fm_index, m) {
    py::class_<FMIndex>(m, "FMIndex")
        .def(py::init<>())  // Default constructor
        .def("get_range", static_cast<std::array<size_t, 3> (FMIndex::*)(py::object, size_t)>(&FMIndex::get_range))
        .def("get_offsets", static_cast<std::vector<size_t> (FMIndex::*)(size_t, size_t)>(&FMIndex::get_offsets))
        .def("load", [](FMIndex& self, const std::string& filename) {
            std::ifstream fin(filename, std::ios::binary);
            if (!fin) {
                throw std::runtime_error("Failed to open file");
            }
            self.load(fin);
        });
}