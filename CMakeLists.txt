cmake_minimum_required(VERSION 3.14)

project(rl-board-game)

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED True)

# ML backends: each one is optional and enabled only when its library is found.
# Targets link rlbg_torch to use it; code checks RLBG_HAS_TORCH / RLBG_TORCH_CUDA.

# LibTorch: set TORCH_ROOT to override the default install locations
set(TORCH_ROOT "" CACHE PATH "LibTorch install prefix")
list(APPEND CMAKE_PREFIX_PATH ${TORCH_ROOT} /opt/libtorch $ENV{HOME}/.local/libtorch)

# a CUDA build of LibTorch needs the CUDA toolkit (nvcc) to configure;
# set CUDA_TOOLKIT_ROOT_DIR to override the default install locations
if(NOT CUDA_TOOLKIT_ROOT_DIR)
  file(GLOB _cuda_local_dirs $ENV{HOME}/.local/cuda-*)
  find_program(_rlbg_nvcc nvcc
    HINTS ENV CUDA_HOME ENV CUDA_PATH
    PATHS /usr/local/cuda ${_cuda_local_dirs}
    PATH_SUFFIXES bin)
  if(_rlbg_nvcc)
    get_filename_component(_cuda_bin_dir ${_rlbg_nvcc} DIRECTORY)
    get_filename_component(CUDA_TOOLKIT_ROOT_DIR ${_cuda_bin_dir} DIRECTORY CACHE)
    set(CMAKE_CUDA_COMPILER ${_rlbg_nvcc} CACHE FILEPATH "")
  endif()
endif()
# nvcc does not support C++26; LibTorch would otherwise inherit CMAKE_CXX_STANDARD
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_HOST_COMPILER ${CMAKE_CXX_COMPILER} CACHE FILEPATH "")
# skip LibTorch's GPU detection; default covers Ampere through Blackwell
set(TORCH_CUDA_ARCH_LIST "8.0;8.6;8.9;9.0;10.0;12.0" CACHE STRING "CUDA architectures for LibTorch")

find_package(Torch QUIET)
if(Torch_FOUND)
  add_library(rlbg_torch INTERFACE)
  target_link_libraries(rlbg_torch INTERFACE ${TORCH_LIBRARIES})
  target_compile_definitions(rlbg_torch INTERFACE RLBG_HAS_TORCH)
  if(CAFFE2_USE_CUDA)
    target_compile_definitions(rlbg_torch INTERFACE RLBG_TORCH_CUDA)
    message(STATUS "ML backend: LibTorch ${Torch_VERSION} with CUDA ${CUDAToolkit_VERSION}")
  else()
    message(STATUS "ML backend: LibTorch ${Torch_VERSION} (CPU only)")
  endif()
else()
  message(STATUS "ML backend: LibTorch not found")
endif()

# MLX: set MLX_ROOT to override the default install locations; the pip package
# (mlx[cuda13]) installs the C++ library under site-packages/mlx
set(MLX_ROOT "" CACHE PATH "MLX install prefix")
file(GLOB _mlx_venv_dirs
  $ENV{VIRTUAL_ENV}/lib/python*/site-packages/mlx
  $ENV{HOME}/.local/mlx/lib/python*/site-packages/mlx)
list(APPEND CMAKE_PREFIX_PATH ${MLX_ROOT} ${_mlx_venv_dirs} /opt/homebrew)

# a CUDA build of MLX needs cuDNN to configure; use the pip one next to it
foreach(_mlx_dir ${_mlx_venv_dirs})
  if(NOT CUDNN_INCLUDE_PATH AND EXISTS ${_mlx_dir}/../nvidia/cudnn/include)
    get_filename_component(CUDNN_INCLUDE_PATH ${_mlx_dir}/../nvidia/cudnn/include ABSOLUTE)
    get_filename_component(CUDNN_LIBRARY_PATH ${_mlx_dir}/../nvidia/cudnn/lib ABSOLUTE)
  endif()
endforeach()

find_package(MLX CONFIG QUIET)
if(MLX_FOUND)
  add_library(rlbg_mlx INTERFACE)
  target_link_libraries(rlbg_mlx INTERFACE ${MLX_LIBRARIES})
  target_compile_definitions(rlbg_mlx INTERFACE RLBG_HAS_MLX)
  if(MLX_BUILD_METAL)
    target_compile_definitions(rlbg_mlx INTERFACE RLBG_MLX_METAL)
    message(STATUS "ML backend: MLX ${MLX_VERSION} with Metal")
  elseif(CUDNN_FOUND)
    target_compile_definitions(rlbg_mlx INTERFACE RLBG_MLX_CUDA)
    message(STATUS "ML backend: MLX ${MLX_VERSION} with CUDA ${CUDAToolkit_VERSION}")
  else()
    message(STATUS "ML backend: MLX ${MLX_VERSION} (CPU only)")
  endif()
else()
  message(STATUS "ML backend: MLX not found")
endif()

include(FetchContent)

set(FTXUI_BUILD_EXAMPLES OFF)
set(FTXUI_BUILD_TESTS    OFF)
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG        v7.0.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ftxui)

set(CLI11_BUILD_TESTS    OFF)
set(CLI11_BUILD_EXAMPLES OFF)
set(CLI11_BUILD_DOCS     OFF)
FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.7.2
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
