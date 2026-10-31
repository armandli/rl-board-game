# game state -> network input; headers are included as <encoder/...>
add_library(encoder INTERFACE)

target_include_directories(encoder INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(encoder INTERFACE game)
