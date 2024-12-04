# connectx_simulator
add_executable(connectx_simulator connectx_simulator.cpp)

set_property(TARGET connectx_simulator PROPERTY CXX_STANDARD 23)

target_link_libraries(connectx_simulator util)
target_link_libraries(connectx_simulator game)
target_link_libraries(connectx_simulator ${CURSES_LIBRARY})

# play connectx against minimax
add_executable(connectx_minimax connectx_minimax.cpp)

set_property(TARGET connectx_minimax PROPERTY CXX_STANDARD 23)

target_link_libraries(connectx_minimax util)
target_link_libraries(connectx_minimax game)
target_link_libraries(connectx_minimax ${CURSES_LIBRARY})

# connectx minimax self play
add_executable(connectx_minimax_selfplay connectx_minimax_selfplay.cpp)

set_property(TARGET connectx_minimax_selfplay PROPERTY CXX_STANDARD 23)

target_link_libraries(connectx_minimax_selfplay util)
target_link_libraries(connectx_minimax_selfplay game)
target_link_libraries(connectx_minimax_selfplay ${CURSES_LIBRARY})


# othello_simulator
add_executable(othello_simulator othello_simulator.cpp)

set_property(TARGET othello_simulator PROPERTY CXX_STANDARD 23)

target_link_libraries(othello_simulator util)
target_link_libraries(othello_simulator game)
target_link_libraries(othello_simulator ${CURSES_LIBRARY})

