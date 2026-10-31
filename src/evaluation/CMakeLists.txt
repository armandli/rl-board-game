# agents, arena, Elo and snapshot evaluation; headers are included as
# <evaluation/...>
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(evaluation INTERFACE)

target_include_directories(evaluation INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(evaluation INTERFACE model search)
