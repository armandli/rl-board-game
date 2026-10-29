# play connectx against minimax
add_executable(connectx_minimax connectx_minimax.cpp)

set_property(TARGET connectx_minimax PROPERTY CXX_STANDARD 26)

target_link_libraries(connectx_minimax util game
  ftxui::screen ftxui::dom ftxui::component)

# play connectx against mcgs
add_executable(connectx_mcgs connectx_mcgs.cpp)

set_property(TARGET connectx_mcgs PROPERTY CXX_STANDARD 26)

target_link_libraries(connectx_mcgs util game
  ftxui::screen ftxui::dom ftxui::component)

# connectx minimax self play
add_executable(connectx_minimax_selfplay connectx_minimax_selfplay.cpp)

set_property(TARGET connectx_minimax_selfplay PROPERTY CXX_STANDARD 26)

target_link_libraries(connectx_minimax_selfplay util)
target_link_libraries(connectx_minimax_selfplay game)

# play othello against mcgs
add_executable(othello_mcgs othello_mcgs.cpp)

set_property(TARGET othello_mcgs PROPERTY CXX_STANDARD 26)

target_link_libraries(othello_mcgs util game
  ftxui::screen ftxui::dom ftxui::component)

# othello mcgs self play
add_executable(othello_mcgs_selfplay othello_mcgs_selfplay.cpp)

set_property(TARGET othello_mcgs_selfplay PROPERTY CXX_STANDARD 26)

target_link_libraries(othello_mcgs_selfplay util)
target_link_libraries(othello_mcgs_selfplay game)

# human vs human connectx simulator
add_executable(connectx_simulator connectx_simulator.cpp)

set_property(TARGET connectx_simulator PROPERTY CXX_STANDARD 26)

target_link_libraries(connectx_simulator util game
  ftxui::screen ftxui::dom ftxui::component)

# human vs human othello simulator
add_executable(othello_simulator othello_simulator.cpp)

set_property(TARGET othello_simulator PROPERTY CXX_STANDARD 26)

target_link_libraries(othello_simulator util game
  ftxui::screen ftxui::dom ftxui::component)

# report the available ML backend and device
if(TARGET rlbg_torch)
  add_executable(torch_info torch_info.cpp)

  set_property(TARGET torch_info PROPERTY CXX_STANDARD 26)

  target_link_libraries(torch_info rlbg_torch)
endif()

if(TARGET rlbg_mlx)
  add_executable(mlx_info mlx_info.cpp)

  set_property(TARGET mlx_info PROPERTY CXX_STANDARD 26)

  target_link_libraries(mlx_info rlbg_mlx)
endif()
