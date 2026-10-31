if(NOT TARGET evaluation)
  return()
endif()

rlbg_add_test(evaluation_test
  SOURCES
    elo_test.cpp
    arena_test.cpp
    snapshot_evaluator_test.cpp
  LIBS evaluation model encoder
  MLX
)
