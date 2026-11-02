if(NOT TARGET rlbg_libtorch)
  return()
endif()

rlbg_add_test(libtorch_test
  SOURCES libtorch_test.cpp
  LIBS rlbg_libtorch encoder
)
