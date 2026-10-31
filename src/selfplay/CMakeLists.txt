# self-play actors, reanalyze and the batching inference server; needs MLX
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(selfplay STATIC
  inference_server.cpp
)

set_property(TARGET selfplay PROPERTY CXX_STANDARD 26)

target_include_directories(selfplay PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(selfplay PUBLIC model experience encoder search)
