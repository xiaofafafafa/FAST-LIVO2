#include <torch/extension.h>
#include "rasterize_points.cuh"
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
  m.def("forward", &RasterizeGaussiansCUDA);
  m.def("backward", &RasterizeGaussiansBackwardCUDA);
}
