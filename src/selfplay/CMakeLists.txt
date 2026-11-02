# self-play and reanalyze actors, trajectory recording and the batching
# inference server, independent of the ML framework; headers are included as
# <selfplay/...>
find_package(Threads REQUIRED)

add_library(selfplay_core INTERFACE)

target_include_directories(selfplay_core INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(selfplay_core INTERFACE experience encoder search Threads::Threads)

# the inference server over an MLX model
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(selfplay INTERFACE)

target_link_libraries(selfplay INTERFACE selfplay_core model)
