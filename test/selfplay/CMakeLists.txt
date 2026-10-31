if(NOT TARGET selfplay)
  return()
endif()

rlbg_add_test(selfplay_test
  SOURCES
    recorder_test.cpp
    inference_server_test.cpp
    actor_test.cpp
  LIBS selfplay
  MLX
)
