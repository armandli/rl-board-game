# framework independent agents
rlbg_add_test(evaluation_core_test
  SOURCES othello_alpha_beta_test.cpp
  LIBS evaluation_core
)

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
