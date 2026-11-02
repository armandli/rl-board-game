# GoogleTest suites, one executable per component; run with `make test`
include(FetchContent)

set(BUILD_GMOCK   OFF CACHE BOOL "" FORCE)
set(INSTALL_GTEST OFF CACHE BOOL "" FORCE)
FetchContent_Declare(googletest
  GIT_REPOSITORY https://github.com/google/googletest.git
  GIT_TAG        v1.17.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(googletest)
include(GoogleTest)

# helpers shared by the suites, included as <common/...>
add_library(test_common INTERFACE)
target_include_directories(test_common INTERFACE ${CMAKE_CURRENT_SOURCE_DIR})

if(TARGET rlbg_mlx)
  # gtest main that runs MLX on the GPU when one is available
  add_library(test_mlx_main STATIC common/mlx_main.cpp)
  set_property(TARGET test_mlx_main PROPERTY CXX_STANDARD 26)
  target_link_libraries(test_mlx_main PUBLIC GTest::gtest rlbg_mlx)
endif()

# rlbg_add_test(<name> SOURCES <files...> LIBS <targets...> [MLX] [LABELS <labels...>])
# MLX suites use test_mlx_main, the others gtest_main
function(rlbg_add_test name)
  cmake_parse_arguments(T "MLX" "" "SOURCES;LIBS;LABELS" ${ARGN})
  add_executable(${name} ${T_SOURCES})
  set_property(TARGET ${name} PROPERTY CXX_STANDARD 26)
  if(T_MLX)
    target_link_libraries(${name} test_common test_mlx_main ${T_LIBS})
  else()
    target_link_libraries(${name} test_common GTest::gtest_main ${T_LIBS})
  endif()
  set(labels ${name} ${T_LABELS})
  gtest_discover_tests(${name}
    DISCOVERY_MODE PRE_TEST
    DISCOVERY_TIMEOUT 60
    PROPERTIES LABELS "${labels}"
  )
endfunction()

add_subdirectory(game)
add_subdirectory(encoder)
add_subdirectory(experience)
add_subdirectory(mlx_util)
add_subdirectory(model)
add_subdirectory(search)
add_subdirectory(selfplay)
add_subdirectory(evaluation)
add_subdirectory(training)
add_subdirectory(libtorch)
