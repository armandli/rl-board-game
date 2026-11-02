# agents, arena, Elo and snapshot evaluation; headers are included as
# <evaluation/...>. Everything but snapshot_evaluator.h (MLX) is independent
# of the ML framework.
add_library(evaluation_core INTERFACE)

target_include_directories(evaluation_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(evaluation_core INTERFACE search)

if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(evaluation INTERFACE)

target_link_libraries(evaluation INTERFACE evaluation_core model)
