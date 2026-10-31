if(NOT TARGET model)
  return()
endif()

rlbg_add_test(model_test
  SOURCES
    transformer_muzero_test.cpp
    model_inference_test.cpp
  LIBS model encoder
  MLX
)
