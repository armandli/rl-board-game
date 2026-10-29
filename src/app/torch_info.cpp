#include <iostream>

#include <torch/torch.h>

namespace s = std;
namespace t = torch;

int main(){
  s::cout << "LibTorch " << TORCH_VERSION << s::endl;
#ifdef RLBG_TORCH_CUDA
  s::cout << "built with CUDA" << s::endl;
#else
  s::cout << "built without CUDA" << s::endl;
#endif

  t::Device device = t::cuda::is_available() ? t::kCUDA : t::kCPU;
  s::cout << "CUDA available: " << (t::cuda::is_available() ? "yes" : "no")
          << ", devices: " << (int)t::cuda::device_count() << s::endl;

  t::Tensor a = t::rand({1024, 1024}, device);
  t::Tensor b = t::rand({1024, 1024}, device);
  t::Tensor c = t::matmul(a, b);
  s::cout << "matmul on " << c.device() << ", sum = " << c.sum().item<float>() << s::endl;
}
