#ifndef RLBG_CONNECTX
#define RLBG_CONNECTX

#include <cassert>
#include <cstdint>
#include <cstring>

#include <algorithm>
#include <array>
#include <bit>
#include <iostream>
#include <ostream>
#include <vector>

#include <type_alias.h>
#include <types.h>
#include <zobrist_hash.h>
#include <game_base.h>

namespace rlbg {

namespace s = std;

template <ubyte R, ubyte C>
struct ConnectXBoard : Board<ConnectXBoard<R,C>> {
  static constexpr uint RZ = R;
  static constexpr uint CZ = C;
  static constexpr uint SZ = R * C;
  // bitboard layout is column major with HZ bits per column; the spare top bit
  // of each column stays empty so lines cannot wrap into the next column
  static constexpr uint HZ = R + 1;
  static_assert(HZ * CZ <= 64, "ConnectX board must fit in a 64 bit bitboard");

  ConnectXBoard(): mBlack(0), mWhite(0) {
    s::memset(mTop.data(), 0U, sizeof(ubyte) * CZ);
  }
  ConnectXBoard(const ConnectXBoard& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mTop = o.mTop;
  }
  ConnectXBoard& operator=(const ConnectXBoard& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mTop = o.mTop;
    return *this;
  }
  ConnectXBoard(ConnectXBoard&& o) noexcept : mBlack(o.mBlack), mWhite(o.mWhite), mTop(s::move(o.mTop)) {}
  ConnectXBoard& operator=(ConnectXBoard&& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mTop = s::move(o.mTop);
    return *this;
  }

  const s::array<ubyte,CZ>& tops() const {
    return mTop;
  }
  uint64_t pieces(Player player) const {
    assert(player != Player::Unknown);
    return player == Player::Black ? mBlack : mWhite;
  }

  bool is_valid(Player player, Pt pt) const {
    if (pt.c >= CZ)       return false;
    if (mTop[pt.c] >= RZ) return false;
    else                  return true;
  }
  Player operator[](Pt pt) const {
    assert(pt.r < RZ and pt.c < CZ);
    uint64_t m = bit(pt.r, pt.c);
    if (mBlack & m) return Player::Black;
    if (mWhite & m) return Player::White;
    return Player::Unknown;
  }
  void place(Player player, Pt pt){
    assert(is_valid(player, pt));
    uint64_t m = bit(mTop[pt.c], pt.c);
    if (player == Player::Black) mBlack |= m;
    else                         mWhite |= m;
    mTop[pt.c] += 1;
  }
  void revert(Pt pt){
    assert(pt.c < CZ);
    ubyte topr = mTop[pt.c];
    assert(topr > 0);
    uint64_t m = bit(topr-1, pt.c);
    mBlack &= ~m;
    mWhite &= ~m;
    mTop[pt.c]--;
  }
  // true if player has X in a row: vertical, horizontal and both diagonals
  template <ubyte X>
  bool has_line(Player player) const {
    uint64_t b = pieces(player);
    for (uint d : {1U, HZ, HZ + 1, HZ - 1}){
      uint64_t m = b;
      for (uint k = 1; k < X and m; ++k)
        m &= k * d < 64 ? b >> (k * d) : 0ULL;
      if (m) return true;
    }
    return false;
  }
  bool is_full() const {
    return (uint)s::popcount(mBlack | mWhite) == SZ;
  }
  size_t size() const {
    return SZ;
  }
  size_t hash() const {
    return mix64(mBlack) ^ mix64(mWhite ^ 0x9e3779b97f4a7c15ULL);
  }
  s::ostream& print(s::ostream& out) const {
    char bchar = 'X';
    char wchar = '0';

    for (int r = RZ-1; r >= 0; --r){
      out << "|";
      for (int c = 0; c < CZ; ++c)
        switch ((*this)[Pt(r, c)]){
        case Player::Black:
          out << bchar;
        break; case Player::White:
          out << wchar;
        break; case Player::Unknown:
          out << ' ';
        break; default: assert(false);
        }
      out << "|\n";
    }
    return out;
  }

private:
  static uint64_t bit(uint r, uint c){
    return 1ULL << (c * HZ + r);
  }

  uint64_t            mBlack;
  uint64_t            mWhite;
  s::array<ubyte, CZ> mTop;
};

template <ubyte R, ubyte C>
s::ostream& operator<<(s::ostream& out, const ConnectXBoard<R, C>& board){
  return board.print(out);
}

template <ubyte RZ, ubyte CZ, ubyte X>
s::pair<bool, Player> compute_termination(const ConnectXBoard<RZ,CZ>& board){
  if (board.template has_line<X>(Player::Black))
    return s::make_pair(true, Player::Black);
  if (board.template has_line<X>(Player::White))
    return s::make_pair(true, Player::White);

  // terminal if board is completely filled
  if (board.is_full())
    return s::make_pair(true, Player::Unknown);
  else
    return s::make_pair(false, Player::Unknown);
}

template <ubyte R, ubyte C, ubyte X>
struct ConnectXGameState : GameState<ConnectXBoard<R,C>, ConnectXGameState<R,C,X>> {
  static constexpr uint RZ = R;
  static constexpr uint CZ = C;
  static constexpr uint SZ = R * C;

  ConnectXGameState()
  : mBoard(), mNPlayer(Player::Black), mHistory(),
    mTerminal(false), mWinner(Player::Unknown)
  {}
  ConnectXGameState(
      const ConnectXBoard<R,C>& board,
      Player nxt_player,
      const s::vector<PlayerMove>& history)
  : mBoard(board), mNPlayer(nxt_player), mHistory(history),
    mTerminal(false), mWinner(Player::Unknown)
  {
    run_test_termination();
  }
  ConnectXGameState(const ConnectXGameState& o)
  : mBoard(o.mBoard),
    mNPlayer(o.mNPlayer),
    mHistory(o.mHistory),
    mTerminal(o.mTerminal),
    mWinner(o.mWinner)
  {}
  ConnectXGameState& operator=(const ConnectXGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mHistory = o.mHistory;
    mTerminal = o.mTerminal;
    mWinner = o.mWinner;
    return *this;
  }
  ConnectXGameState(ConnectXGameState&& o)
  : mBoard(s::move(o.mBoard)),
    mNPlayer(o.mNPlayer),
    mHistory(s::move(o.mHistory)),
    mTerminal(o.mTerminal),
    mWinner(o.mWinner)
  {}
  ConnectXGameState& operator=(ConnectXGameState&& o){
    //TODO: correct ?
    mBoard    = s::move(o.mBoard);
    mNPlayer  = o.mNPlayer;
    mHistory  = s::move(o.mHistory);
    mTerminal = o.mTerminal;
    mWinner   = o.mWinner;
    return *this;
  }

  const ConnectXBoard<R,C>& board() const {
    return mBoard;
  }
  Player next_player() const {
    return mNPlayer;
  }
  bool is_valid(PlayerMove pm) const {
    if (is_terminal()) return false;
    switch (pm.move.mty){
    case M::Play:
      return mBoard.is_valid(pm.player, pm.move.mpt);
    break; case M::Pass: case M::Resign: case M::Unknown:
      return false;
    break; default: assert(false); // should never get here
    }
    return false;
  }
  MoveList<CZ> legal_moves() const {
    MoveList<CZ> moves;
    if (is_terminal()) return moves;
    for (int c = 0; c < CZ; ++c)
      if (mBoard.is_valid(mNPlayer, Pt(0,c)))
        moves.push_back(Move(M::Play, Pt(0,c)));
    return moves;
  }
  bool apply(PlayerMove pm){
    if (is_terminal()) return false;
    if (not is_valid(pm)) return false;
    mBoard.place(pm.player, pm.move.mpt);
    mHistory.push_back(pm);
    mNPlayer = opponent(mNPlayer);
    run_test_termination();
    return true;
  }
  void rollback(int steps){
    // the bound is fixed up front: the loop shrinks the history
    size_t n = s::min((size_t)steps, mHistory.size());
    for (size_t i = 0; i < n; ++i){
      PlayerMove pm = mHistory.back();
      mHistory.pop_back();
      if (pm.move.mty == M::Play)
        mBoard.revert(pm.move.mpt);
    }
    if (mHistory.size() > 0) mNPlayer = opponent(mHistory.back().player);
    else                     mNPlayer = Player::Black;
    // apply() refuses to move from a terminal state, so every earlier state
    // in the history was still in play
    if (n > 0){
      mTerminal = false;
      mWinner = Player::Unknown;
    }
  }
  bool is_terminal() const {
    return mTerminal;
  }
  Player winner() const {
    return mWinner;
  }
  int score(Player player) const {
    if (not is_terminal()) return 0;
    Player w = winner();
    if (w == Player::Unknown) return 0;
    else if (w == player)     return 1;
    else                      return -1;
  }
  size_t hash() const {
    size_t h = mBoard.hash();
    h ^= zobrist_hash((uint)mNPlayer);
    return h;
  }
  const s::vector<PlayerMove>& history() const {
    return mHistory;
  }

protected:
  void run_test_termination(){
    auto [is_terminal, winner] = compute_termination<R,C,X>(mBoard);
    mTerminal = is_terminal;
    mWinner = winner;
  }

  ConnectXBoard<R, C>   mBoard;
  Player                mNPlayer;
  s::vector<PlayerMove> mHistory;
  bool                  mTerminal;
  Player                mWinner;
};

}  // namespace rlbg

namespace r = rlbg;

namespace std {
template <ubyte R, ubyte C>
struct hash<r::ConnectXBoard<R,C>> {
  size_t operator()(const r::ConnectXBoard<R,C>& board) const {
    return board.hash();
  }
};
template <ubyte R, ubyte C, ubyte X>
struct hash<r::ConnectXGameState<R,C,X>> {
  size_t operator()(const r::ConnectXGameState<R,C,X>& state) const {
    return state.hash();
  }
};
}  // namespace std

#endif//RLBG_CONNECTX
