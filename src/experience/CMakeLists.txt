# trajectories, replay buffer and on-disk experience store; headers are
# included as <experience/...>
add_library(experience INTERFACE)

target_include_directories(experience INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)
