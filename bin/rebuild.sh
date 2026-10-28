#!/bin/sh

NCURSES_DEV=/tmp/ncurses-dev

# Extract ncurses dev package locally if headers are missing
if [ ! -f "$NCURSES_DEV/usr/include/curses.h" ]; then
  apt-get download libncurses-dev
  mkdir -p "$NCURSES_DEV"
  dpkg-deb -x libncurses-dev_*.deb "$NCURSES_DEV"
  rm -f libncurses-dev_*.deb
fi

rm -rf build 2>/dev/null
mkdir build
cd build

cmake \
  -DCURSES_INCLUDE_PATH="$NCURSES_DEV/usr/include" \
  "-DCURSES_LIBRARY=$NCURSES_DEV/usr/lib/x86_64-linux-gnu/libncurses.a;$NCURSES_DEV/usr/lib/x86_64-linux-gnu/libtinfo.a" \
  -DCMAKE_INSTALL_PREFIX=install \
  ..

cmake --build . --config Release

cd ..
