#!/bin/sh

rm -rf build 2>/dev/null
mkdir build
cd build

cmake \
  -DCMAKE_INSTALL_PREFIX=install \
  ..

cmake --build . --config Release

cd ..
