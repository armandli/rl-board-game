# MuZero networks; needs MLX
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(model STATIC
  model_inference.cpp
  transformer_muzero.cpp
)

set_property(TARGET model PROPERTY CXX_STANDARD 26)

target_include_directories(model PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(model PUBLIC mlx_util search)
