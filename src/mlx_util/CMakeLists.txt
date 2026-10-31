# C++ counterpart of Python's mlx.nn / mlx.optimizers; needs MLX
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(mlx_util STATIC
  activations.cpp
  containers.cpp
  convolution.cpp
  dropout.cpp
  embedding.cpp
  init.cpp
  io.cpp
  linear.cpp
  losses.cpp
  module.cpp
  normalization.cpp
  optimizers.cpp
  pooling.cpp
  positional_encoding.cpp
  recurrent.cpp
  schedulers.cpp
  transformer.cpp
  upsample.cpp
)

set_property(TARGET mlx_util PROPERTY CXX_STANDARD 26)

# headers are included as <mlx_util/...>
target_include_directories(mlx_util PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(mlx_util PUBLIC rlbg_mlx util)

# zlib reads and writes compressed .npz (mx.savez_compressed / np.savez_compressed)
find_package(ZLIB QUIET)
if(ZLIB_FOUND)
  target_link_libraries(mlx_util PRIVATE ZLIB::ZLIB)
  target_compile_definitions(mlx_util PRIVATE RLBG_HAS_ZLIB)
else()
  message(STATUS "mlx_util: zlib not found, compressed .npz unsupported")
endif()

add_subdirectory(test)
