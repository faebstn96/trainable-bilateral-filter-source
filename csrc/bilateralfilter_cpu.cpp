#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_3d_cpu", &BilateralFilterCpuForward, "BF forward 3d cpu");
    m.def("backward_3d_cpu", &BilateralFilterCpuBackward, "BF backward 3d cpu");
}
