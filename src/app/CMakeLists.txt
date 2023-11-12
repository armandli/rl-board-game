# connectx_simulator
add_executable(connectx_simulator connectx_simulator.cpp)

set_property(TARGET connectx_simulator PROPERTY CXX_STANDARD 20)

target_link_libraries(connectx_simulator util)
target_link_libraries(connectx_simulator game)

# othello_simulator
add_executable(othello_simulator othello_simulator.cpp)

set_property(TARGET othello_simulator PROPERTY CXX_STANDARD 20)

target_link_libraries(othello_simulator util)
target_link_libraries(othello_simulator game)

