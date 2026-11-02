# MuZero on LibTorch: transformer model, batched GPU inference, learner and
# the training backend; headers are included as <libtorch/...>
if(NOT TARGET rlbg_torch)
  return()
endif()

add_library(rlbg_libtorch STATIC
  learner.cpp
  model_inference.cpp
  transformer_muzero.cpp
)

set_property(TARGET rlbg_libtorch PROPERTY CXX_STANDARD 26)

target_include_directories(rlbg_libtorch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(rlbg_libtorch PUBLIC rlbg_torch training_core)
