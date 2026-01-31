#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "../cpp/kernels.h"

namespace py = pybind11;

PYBIND11_MODULE(bitinfer_cpp, m) {
    m.doc() = "BitInfer C++ acceleration kernels";

    m.def("quantized_matmul", [](
        py::array_t<float> input,
        py::array_t<int8_t> weight,
        py::array_t<float> output,
        int batch_size,
        int input_dim,
        int output_dim
    ) {
        quantized_matmul(
            static_cast<float*>(input.mutable_unchecked().mutable_data()),
            static_cast<int8_t*>(weight.mutable_unchecked().mutable_data()),
            static_cast<float*>(output.mutable_unchecked().mutable_data()),
            batch_size, input_dim, output_dim
        );
    }, "Fast quantized matrix multiplication");

    m.def("batch_quantize", [](
        py::array_t<float> input,
        py::array_t<int8_t> output,
        py::array_t<float> scale,
        int size
    ) {
        batch_quantize(
            static_cast<float*>(input.mutable_unchecked().mutable_data()),
            static_cast<int8_t*>(output.mutable_unchecked().mutable_data()),
            static_cast<float*>(scale.mutable_unchecked().mutable_data()),
            size
        );
    }, "Batch quantization");
}
