#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include "fm_index2.hpp"  // 包含你的FMIndex类的头文件

namespace py = pybind11;
using istring_view = std::basic_string_view<signed char>;

template<int SA_INTV, typename size_type, typename Sorter>
std::vector<size_type> get_range_wrapper(biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, const std::string &seed, size_type stop_cnt) {
    auto converted_seed = biovoltron::Codec::to_istring(seed);
    // const signed char* seed_data = reinterpret_cast<const signed char*>(seed.data());
    istring_view seed_view(converted_seed.data(), converted_seed.size());
    
    // 获取 std::array
    auto range = fmi.get_range(seed_view, stop_cnt);

    // 将 std::array 转换为 std::vector
    std::vector<size_type> range_vector(range.begin(), range.end());
    return range_vector;
}

template<int SA_INTV, typename size_type, typename Sorter>
std::vector<size_type> get_offsets_wrapper(biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, size_type beg, size_type end) {
    auto span_result = fmi.get_offsets(beg, end);
    return std::vector<size_type>(span_result.begin(), span_result.end());
}

template<int SA_INTV, typename size_type, typename Sorter>
void load_wrapper(biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, const std::string &filename) {
    std::ifstream fin(filename, std::ios::binary);
    if (!fin) {
        throw std::runtime_error("Failed to open file: " + filename);
    }
    fmi.load(fin);
}

template<int SA_INTV, typename size_type, typename Sorter>
void bind_FMIndex(py::module &m, const std::string &class_name) {
    py::class_<biovoltron::FMIndex<SA_INTV, size_type, Sorter>>(m, class_name.c_str())
        .def(py::init<>())
        .def(py::init<const std::unordered_set<int>&>(),  // 新增带 end_positions 初始化的构造函数
            py::arg("end_positions"))
        .def("load", [](biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, const std::string &filename) {
            load_wrapper(fmi, filename);
        })
        .def("get_offsets", [](biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, size_type beg, size_type end) {
            return get_offsets_wrapper(fmi, beg, end);
        })
        .def("get_range", [](biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, const std::string &seed, size_type stop_cnt = 0) {
            return get_range_wrapper(fmi, seed, stop_cnt);
        })
        .def("create_pseudogene", [](biovoltron::FMIndex<SA_INTV, size_type, Sorter> &fmi, const std::vector<std::string>& sequences, const std::string& output_file) {
            fmi.create_pseudogene(sequences, output_file);
        })
        .def_readwrite("end_positions", &biovoltron::FMIndex<SA_INTV, size_type, Sorter>::end_positions)
        .def("__eq__", &biovoltron::FMIndex<SA_INTV, size_type, Sorter>::operator==);
}

PYBIND11_MODULE(fm_index, m) {
    // 这里假设你有不同的模板参数组合，具体根据你需要的绑定模板实例组合

    bind_FMIndex<4, std::uint32_t, biovoltron::KISS1Sorter<std::uint32_t>>(m, "FMIndex_Uint32_KISS1");
    // 你可以根据需要绑定不同的模板实例
}