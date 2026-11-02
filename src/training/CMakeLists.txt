# training batches, sample ratio control and the training loop, independent
# of the ML framework; headers are included as <training/...>
add_library(training_core INTERFACE)

target_include_directories(training_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(training_core INTERFACE selfplay_core evaluation_core experience encoder)

# learner and training loop of MLX models
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(training STATIC
  muzero_learner.cpp
)

set_property(TARGET training PROPERTY CXX_STANDARD 26)

target_include_directories(training PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(training PUBLIC training_core model selfplay evaluation)
