if(NOT TARGET search)
  return()
endif()

rlbg_add_test(search_test
  SOURCES mcgs_test.cpp
  LIBS model evaluation search encoder
  MLX
)
