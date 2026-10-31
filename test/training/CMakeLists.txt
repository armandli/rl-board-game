if(NOT TARGET training)
  return()
endif()

rlbg_add_test(training_test
  SOURCES
    sample_ratio_test.cpp
    batch_test.cpp
    learner_test.cpp
    training_loop_test.cpp
  LIBS training
  MLX
)
