# planning search over a learned model; needs MLX for the half precision latent type
if(NOT TARGET rlbg_mlx)
  return()
endif()

add_library(search INTERFACE)

target_include_directories(search INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(search INTERFACE rlbg_mlx encoder game)
