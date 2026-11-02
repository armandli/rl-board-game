# planning search over a learned model, independent of the ML framework;
# headers are included as <search/...>
add_library(search INTERFACE)

target_include_directories(search INTERFACE ${CMAKE_CURRENT_SOURCE_DIR}/..)

target_link_libraries(search INTERFACE encoder game)
