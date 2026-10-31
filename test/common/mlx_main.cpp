// gtest entry point for suites that use MLX: runs on the GPU when available
#include <gtest/gtest.h>
#include <mlx/mlx.h>

namespace mx = mlx::core;

int main(int argc, char** argv){
  testing::InitGoogleTest(&argc, argv);
  mx::Device gpu(mx::Device::gpu);
  if (mx::is_available(gpu)) mx::set_default_device(gpu);
  return RUN_ALL_TESTS();
}
