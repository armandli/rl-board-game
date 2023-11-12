#ifndef RLBG_TYPES
#define RLBG_TYPES

#include <cassert>
#include <array>
#include <functional>
#include <ostream>

#include <type_alias.h>

namespace rlbg {

namespace s = std;

struct Direction {
  byte r, c;

  Direction() = default;
  constexpr Direction(byte r, byte c) noexcept: r(r), c(c) {}
};

struct Pt {
  ubyte r, c;

  Pt() = default;
  Pt(ubyte r, ubyte c): r(r), c(c) {}

  Pt& operator+=(Direction d){
    r += d.r;
    c += d.c;
    return *this;
  }

  Pt& operator-=(Direction d){
    r -= d.r;
    c -= d.c;
    return *this;
  }
};

using Neighbours4 = s::array<Pt, 4>;
using Neighbours8 = s::array<Pt, 8>;

bool operator==(Pt a, Pt b){
  return a.r == b.r && a.c == b.c;
}

bool operator!=(Pt a, Pt b){
  return not operator==(a, b);
}

template <uint SZ>
[[gnu::always_inline]] uint index(Pt p){
  return p.r * SZ + p.c;
}

template <uint SZ>
[[gnu::always_inline]] Pt point(uint idx){
  return Pt(idx / SZ, idx % SZ);
}


bool operator==(Direction a, Direction b){
  return a.r == b.r && a.c == b.c;
}

bool operator!=(Direction a, Direction b){
  return not operator==(a, b);
}

Pt operator+(Pt pt, Direction d){
  return Pt(pt.r+d.r, pt.c+d.c);
}

Pt operator-(Pt pt, Direction d){
  return Pt(pt.r-d.r, pt.c-d.c);
}

constexpr s::array<Direction, 8> EIGHT_WAY {
  Direction( 1, 0),
  Direction( 1, 1),
  Direction( 0, 1),
  Direction(-1, 1),
  Direction(-1, 0),
  Direction(-1,-1),
  Direction( 0,-1),
  Direction( 1,-1),
};

[[gnu::always_inline]] Neighbours4 neighbours(Pt p){
  return Neighbours4{
    Pt(p.r-1, p.c),
    Pt(p.r,   p.c-1),
    Pt(p.r+1, p.c),
    Pt(p.r,   p.c+1),
  };
}

[[gnu::always_inline]] Neighbours4 corners(Pt p){
  return Neighbours4{
    Pt(p.r-1, p.c-1),
    Pt(p.r+1, p.c-1),
    Pt(p.r-1, p.c+1),
    Pt(p.r+1, p.c+1),
  };
}

[[gnu::always_inline]] Neighbours8 surroundings(Pt p){
  return Neighbours8{
    Pt(p.r-1, p.c),
    Pt(p.r-1, p.c+1),
    Pt(p.r,   p.c+1),
    Pt(p.r+1, p.c+1),
    Pt(p.r+1, p.c),
    Pt(p.r+1, p.c-1),
    Pt(p.r,   p.c-1),
    Pt(p.r-1, p.c-1),
  };
}

enum class M : ubyte {
  Play    = 0x1,
  Pass    = 0x2,
  Resign  = 0x3,
  Unknown = 0x0,
};

struct Move {
  M  mty;
  Pt mpt;

  explicit Move(M ty): mty(ty) {
    assert(mty != M::Play);
  }
  Move(M ty, Pt p): mty(ty), mpt(p) {}
  Move(const Move& o): mty(o.mty), mpt(o.mpt) {}
  Move& operator=(const Move& o){
    mty = o.mty;
    mpt = o.mpt;
    return *this;
  }
};

bool operator==(const Move& a, const Move& b){
  switch (a.mty){
  case M::Play:
    return a.mty == b.mty && a.mpt == b.mpt;
  case M::Pass:
  case M::Resign:
  case M::Unknown:
    return a.mty == b.mty;
  default: assert(false);
  }
}

bool operator!=(const Move& a, const Move& b){
  return not operator==(a, b);
}

enum class Player: ubyte {
  Black   = 0x1,
  White   = 0x2,
  Unknown = 0x3, //can used to represent empty
};

[[gnu::always_inline]] Player opponent(Player player){
  return (Player)((ubyte)player ^ 0x3U);
}

s::ostream& operator<<(s::ostream& out, Player player){
  switch (player){
  break; case Player::Black:
    out << "Black";
  break; case Player::White:
    out << "White";
  break; case Player::Unknown:
    out << "Tie";
  break; default: assert(false);
  }
  return out;
}

struct PlayerMove {
  Player player;
  Move   move;

  PlayerMove(): player(Player::Unknown), move(M::Unknown) {}
  PlayerMove(Player p, Move m): player(p), move(m) {}
  PlayerMove(const PlayerMove& o): player(o.player), move(o.move) {}
};

bool operator==(const PlayerMove& a, const PlayerMove& b){
  return a.player == b.player && a.move == b.move;
}

bool operator!=(const PlayerMove& a, const PlayerMove& b){
  return not operator==(a, b);
}

} // rlbg

namespace r = rlbg;

namespace std {
template <>
struct hash<r::Pt> {
  size_t operator()(const r::Pt& pt) const {
    return (pt.r << 8U) + pt.c;
  }
};
} // std

#endif//RLBG_TYPES
