#ifndef RLBG_TYPES
#define RLBG_TYPES

#include <cassert>
#include <ostream>

#include <type_alias.h>

namespace rlbg {

namespace s = std;

struct Pt {
  ubyte r, c;

  Pt() = default;
  Pt(ubyte r, ubyte c): r(r), c(c) {}
};

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

//TODO: may also need skip ?
enum class M : ubyte {
  Play    = 0x1,
  Resign  = 0x2,
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
  Unknown = 0x3,
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

#endif//RLBG_TYPES
