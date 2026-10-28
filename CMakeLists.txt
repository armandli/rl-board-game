cmake_minimum_required(VERSION 3.14)

project(rl-board-game)

list(APPEND CMAKE_PREFIX_PATH "/opt/libtorch")

set(CMAKE_CXX_STANDARD 26)
set(CMAKE_CXX_STANDARD_REQUIRED True)

find_package(Torch QUIET)
if(Torch_FOUND)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")
endif()

include(FetchContent)

set(FTXUI_BUILD_EXAMPLES OFF)
set(FTXUI_BUILD_TESTS    OFF)
FetchContent_Declare(ftxui
  GIT_REPOSITORY https://github.com/ArthurSonzogni/FTXUI
  GIT_TAG        v7.0.0
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(ftxui)

set(CLI11_BUILD_TESTS    OFF)
set(CLI11_BUILD_EXAMPLES OFF)
set(CLI11_BUILD_DOCS     OFF)
FetchContent_Declare(cli11
  GIT_REPOSITORY https://github.com/CLIUtils/CLI11.git
  GIT_TAG        v2.7.2
  GIT_SHALLOW    TRUE
)
FetchContent_MakeAvailable(cli11)

set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/lib)
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

add_subdirectory(src)
