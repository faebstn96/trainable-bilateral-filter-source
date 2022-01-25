#include "bilateral.h"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("forward_3d_gpu", &BilateralFilterCudaForward, "BF forward 3d gpu");
    m.def("backward_3d_gpu", &BilateralFilterCudaBackward, "BF backward 3d gpu");
}
