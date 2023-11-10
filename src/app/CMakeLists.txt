# connectx_simulator
add_executable(connectx_simulator connectx_simulator.cpp)

set_property(TARGET connectx_simulator PROPERTY CXX_STANDARD 17)

target_link_libraries(connectx_simulator util)
target_link_libraries(connectx_simulator game)

# reversi_simulator
add_executable(reversi_simulator reversi_simulator.cpp)

set_property(TARGET reversi_simulator PROPERTY CXX_STANDARD 17)

target_link_libraries(reversi_simulator util)
target_link_libraries(reversi_simulator game)

