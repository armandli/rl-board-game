#include <iostream>

#include <mlx/mlx.h>

namespace s = std;
namespace mx = mlx::core;

int main(){
  s::cout << "MLX " << mx::version() << s::endl;
#if defined(RLBG_MLX_CUDA)
  s::cout << "built with CUDA" << s::endl;
#elif defined(RLBG_MLX_METAL)
  s::cout << "built with Metal" << s::endl;
#else
  s::cout << "built without GPU" << s::endl;
#endif

  mx::Device gpu(mx::Device::gpu);
  bool has_gpu = mx::is_available(gpu);
  s::cout << "GPU available: " << (has_gpu ? "yes" : "no") << s::endl;

  mx::Device device = has_gpu ? gpu : mx::Device(mx::Device::cpu);
  mx::set_default_device(device);

  mx::array a = mx::random::uniform({1024, 1024});
  mx::array b = mx::random::uniform({1024, 1024});
  mx::array c = mx::sum(mx::matmul(a, b));
  mx::eval(c);
  s::cout << "matmul on " << (device == gpu ? "gpu" : "cpu") << ", sum = " << c.item<float>() << s::endl;
}
