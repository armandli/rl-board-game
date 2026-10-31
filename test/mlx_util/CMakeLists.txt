if(NOT TARGET mlx_util)
  return()
endif()

rlbg_add_test(mlx_util_test
  SOURCES mlx_util_test.cpp zoo.cpp
  LIBS mlx_util
  MLX
)

# C++ side of the Python MLX compatibility test:
#   python test/mlx_util/python_compat.py build/bin/mlx_util_compat
add_executable(mlx_util_compat mlx_util_compat.cpp zoo.cpp)

set_property(TARGET mlx_util_compat PROPERTY CXX_STANDARD 26)

target_link_libraries(mlx_util_compat mlx_util)
