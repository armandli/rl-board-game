# learner and the reinforcement learning training loop; needs MLX
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(training STATIC
  muzero_learner.cpp
)

set_property(TARGET training PROPERTY CXX_STANDARD 26)

target_include_directories(training PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(training PUBLIC model selfplay evaluation experience encoder)

