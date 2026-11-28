# MLX suite: the search tests run against the MLX true_model fixture
if(NOT TARGET model)
  return()
endif()

rlbg_add_test(search_test
  SOURCES mcgs_test.cpp
  LIBS model evaluation search encoder
  MLX
)
