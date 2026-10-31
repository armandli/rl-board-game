rlbg_add_test(game_test
  SOURCES
    types_test.cpp
    othello_test.cpp
    othello_bitboard_test.cpp
    connectx_test.cpp
    classic_search_test.cpp
  LIBS util game
)
