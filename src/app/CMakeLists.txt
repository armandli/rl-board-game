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

# MuZero self-play training for othello
if(TARGET training)
  add_executable(othello_mcgs_selfplay_train othello_mcgs_selfplay_train.cpp)

  set_property(TARGET othello_mcgs_selfplay_train PROPERTY CXX_STANDARD 26)

  target_link_libraries(othello_mcgs_selfplay_train training CLI11::CLI11)
endif()

# MuZero self-play training, arena and play for othello on LibTorch
if(TARGET rlbg_libtorch)
  add_executable(othello_muzero_torch_train othello_muzero_torch_train.cpp)

  set_property(TARGET othello_muzero_torch_train PROPERTY CXX_STANDARD 26)

  target_link_libraries(othello_muzero_torch_train rlbg_libtorch CLI11::CLI11)
endif()

if(TARGET rlbg_libtorch)
  add_executable(othello_muzero_torch_bench othello_muzero_torch_bench.cpp)

  set_property(TARGET othello_muzero_torch_bench PROPERTY CXX_STANDARD 26)

  target_link_libraries(othello_muzero_torch_bench rlbg_libtorch CLI11::CLI11)
endif()

if(TARGET rlbg_libtorch)
  add_executable(othello_muzero_torch_arena othello_muzero_torch_arena.cpp)

  set_property(TARGET othello_muzero_torch_arena PROPERTY CXX_STANDARD 26)

  target_link_libraries(othello_muzero_torch_arena rlbg_libtorch CLI11::CLI11)
endif()

if(TARGET rlbg_libtorch)
  add_executable(othello_muzero_torch_play othello_muzero_torch_play.cpp)

  set_property(TARGET othello_muzero_torch_play PROPERTY CXX_STANDARD 26)

  target_link_libraries(othello_muzero_torch_play rlbg_libtorch CLI11::CLI11
    ftxui::screen ftxui::dom ftxui::component)
endif()
