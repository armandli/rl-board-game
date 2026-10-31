BUILD_DIR  ?= build
BUILD_TYPE ?= Release

# CMake only enables C++26 for g++ >= 14 / clang++ >= 17; pick the first one available
CXX_CANDIDATES := g++-16 g++-15 g++-14 clang++-23 clang++-22 clang++-21 clang++-20 clang++-19 clang++-18
# on macOS use Homebrew LLVM: Apple clang lags on C++26, and g++'s libstdc++ cannot link the libc++ built MLX
ifeq ($(shell uname -s),Darwin)
  CXX_CANDIDATES := $(wildcard $(shell brew --prefix llvm 2>/dev/null)/bin/clang++)
endif
ifeq ($(origin CXX),default)
  CXX := $(firstword $(foreach c,$(CXX_CANDIDATES),$(if $(shell command -v $(c)),$(c))))
endif
ifeq ($(origin CC),default)
  # clang++ contains g++, so check it first
  CC := $(if $(findstring clang++,$(CXX)),$(subst clang++,clang,$(CXX)),$(subst g++,gcc,$(CXX)))
endif

.PHONY: all configure rebuild clean

all: $(BUILD_DIR)/CMakeCache.txt
	cmake --build $(BUILD_DIR) --parallel

configure $(BUILD_DIR)/CMakeCache.txt:
	cmake -S . -B $(BUILD_DIR) \
	  -DCMAKE_BUILD_TYPE=$(BUILD_TYPE) \
	  -DCMAKE_INSTALL_PREFIX=install \
	  -DCMAKE_CXX_COMPILER=$(CXX) \
	  -DCMAKE_C_COMPILER=$(CC)

rebuild: clean
	$(MAKE) all

clean:
	rm -rf $(BUILD_DIR)
