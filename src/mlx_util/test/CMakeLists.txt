# unit tests and the C++ side of the Python MLX compatibility test:
#   mlx_util_test unit
#   python src/mlx_util/test/python_compat.py build/bin/mlx_util_test
add_executable(mlx_util_test mlx_util_test.cpp zoo.cpp)

set_property(TARGET mlx_util_test PROPERTY CXX_STANDARD 26)

target_link_libraries(mlx_util_test mlx_util)
