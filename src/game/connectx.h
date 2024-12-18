#ifndef RLBG_CONNECTX
#define RLBG_CONNECTX

#include <cassert>
#include <array>
#include <vector>
#include <algorithm>
#include <ostream>

#include <type_alias.h>
#include <types.h>
#include <zobrist_hash.h>
#include <game_base.h>

#include <iostream>

namespace rlbg {

namespace s = std;

template <ubyte R, ubyte C>
struct ConnectXBoard : Board<ConnectXBoard<R,C>> {
  static constexpr uint RZ = R;
  static constexpr uint CZ = C;
  static constexpr uint SZ = R * C;
private:
  s::array<Player, SZ> mBoard;
  s::array<ubyte,  CZ> mTop;
public:
  ConnectXBoard(){
    s::memset(mBoard.data(), (uint)Player::Unknown, sizeof(ubyte) * SZ);
    s::memset(mTop.data(), 0U, sizeof(ubyte) * CZ);
  }
  ConnectXBoard(const ConnectXBoard& o){
    mBoard = o.mBoard;
    mTop = o.mTop;
  }
  ConnectXBoard& operator=(const ConnectXBoard& o){
    mBoard = o.mBoard;
    mTop = o.mTop;
    return *this;
  }
  ConnectXBoard(ConnectXBoard&& o) noexcept : mBoard(s::move(o.mBoard)) {}
  ConnectXBoard& operator=(ConnectXBoard&& o){
    //TODO: correct ?
    mBoard = s::move(o.mBoard);
    mTop = s::move(o.mTop);
    return *this;
  }

  const s::array<ubyte,CZ>& tops() const {
    return mTop;
  }

  bool is_valid(Player player, Pt pt) const {
    if (pt.c >= CZ)       return false;
    if (mTop[pt.c] >= RZ) return false;
    else                  return true;
  }
  Player operator[](Pt pt) const {
    assert(pt.r < RZ && pt.c < CZ);
    return mBoard[index<CZ>(pt)];
  }
  void place(Player player, Pt pt){
    assert(is_valid(player, pt));
    ubyte topr = mTop[pt.c];
    mBoard[index<CZ>(Pt(topr, pt.c))] = player;
    mTop[pt.c] += 1;
  }
  // assume only reverting the last step
  void revert(Pt pt){
    assert(pt.c < CZ);
    ubyte topr = mTop[pt.c];
    assert(topr > 0);
    mBoard[index<CZ>(Pt(topr-1, pt.c))] = Player::Unknown;
    mTop[pt.c]--;
  }
  size_t size() const {
    return SZ;
  }
  size_t hash() const {
    size_t h = 0;
    for (uint i = 0; i < mBoard.size(); ++i){
      if (mBoard[i] != Player::Unknown)
        h ^= zobrist_hash((uint)mBoard[i], i);
    }
    return h;
  }
  s::ostream& print(s::ostream& out) const {
    char bchar = 'X';
    char wchar = '0';

    for (int r = RZ-1; r >= 0; --r){
      out << "|";
      for (int c = 0; c < CZ; ++c)
        switch (mBoard[index<CZ>(Pt(r, c))]){
        break; case Player::Black:
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
};

template <ubyte R, ubyte C>
s::ostream& operator<<(s::ostream& out, const ConnectXBoard<R, C>& board){
  return board.print(out);
}

//TODO: write unit test
template <ubyte RZ, ubyte CZ, ubyte X>
s::pair<bool, Player> compute_termination(const ConnectXBoard<RZ,CZ>& board){
    // terminal if one player connected X
    int max_count = 0;
    Player max_player = Player::Unknown;
    // NOTE: outter for loops can be parallelized
    // horizontal scan
    for (int r = 0; r < RZ; ++r){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int c = 0; c < CZ; ++c)
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(r,c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(r,c)];
        } else
          count += 1;
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }
    // vertical scan
    for (int c = 0; c < CZ; ++c){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int r = 0; r < RZ; ++r)
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(r,c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(r,c)];
        } else
          count += 1;
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }
    // buttom up diagonal scan
    for (int r = RZ-1; r >= 0; --r){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int c = 0; c + r < RZ && c < CZ; ++c){
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(c+r,c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(c+r,c)];
        } else
          count += 1;
      }
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }
    for (int r = 0; r < RZ; ++r){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int c = CZ-1; r-((int)CZ-1-c) >= 0 && c >= 0; --c){
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(r-(CZ-1-c),c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(r-(CZ-1-c),c)];
        } else 
          count += 1;
      }
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }
    // top down diagonal scan
    for (int r = RZ-1; r >= 0; --r){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int c = 0; r-c >= 0 && c < CZ; ++c){
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(r-c,c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(r-c,c)];
        } else 
          count += 1;
      }
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }
    for (int r = RZ-1; r >= 0; --r){
      int count = 0;
      Player prev_piece = Player::Unknown;
      for (int c = CZ-1; r+((int)CZ-1-c) < RZ && c >= 0; --c){
        if (prev_piece == Player::Unknown || prev_piece != board[Pt(r+(CZ-1-c),c)]){
          if (count > max_count){
            max_count = count;
            max_player = prev_piece;
          }
          count = 1;
          prev_piece = board[Pt(r+(CZ-1-c),c)];
        } else
          count += 1;
      }
      if (count > max_count){
        max_count = count;
        max_player = prev_piece;
      }
    }

    if (max_count >= X)
      return s::make_pair(true, max_player);

    // terminal if board is completely filled
    ubyte min_row = *s::min_element(s::begin(board.tops()), s::end(board.tops()));
    if (min_row == RZ)
      return s::make_pair(true, Player::Unknown);
    else
      return s::make_pair(false, Player::Unknown);
}

template <ubyte R, ubyte C, ubyte X>
struct ConnectXGameState : GameState<ConnectXBoard<R,C>, ConnectXGameState<R,C,X>> {
  static constexpr uint RZ = R;
  static constexpr uint CZ = C;
  static constexpr uint SZ = R * C;
protected:
  ConnectXBoard<R, C>   mBoard;
  Player                mNPlayer;
  s::vector<PlayerMove> mHistory;
  bool                  mTerminal;
  Player                mWinner;
public:
  ConnectXGameState(): mBoard(), mNPlayer(Player::Black), mHistory(), mTerminal(false), mWinner(Player::Unknown) {}
  ConnectXGameState(const ConnectXBoard<R,C>& board, Player nxt_player, const s::vector<PlayerMove>& history): mBoard(board), mNPlayer(nxt_player), mHistory(history), mTerminal(false), mWinner(Player::Unknown) {
    run_test_termination();
  }
  ConnectXGameState(const ConnectXGameState& o): mBoard(o.mBoard), mNPlayer(o.mNPlayer), mHistory(o.mHistory), mTerminal(o.mTerminal), mWinner(o.mWinner) {}
  ConnectXGameState& operator=(const ConnectXGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mHistory = o.mHistory;
    mTerminal = o.mTerminal;
    mWinner = o.mWinner;
    return *this;
  }
  ConnectXGameState(ConnectXGameState&& o): mBoard(s::move(o.mBoard)), mNPlayer(o.mNPlayer), mHistory(s::move(o.mHistory)), mTerminal(o.mTerminal), mWinner(o.mWinner) {}
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
    break; case M::Play:
      return mBoard.is_valid(pm.player, pm.move.mpt);
    break; case M::Pass:
           case M::Resign:
           case M::Unknown:
      return false;
    }
  }
  s::vector<Move> legal_moves() const {
    if (is_terminal()) return s::vector<Move>();
    s::vector<Move> moves;
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
    for (size_t i = 0; i < s::min((size_t)steps, mHistory.size()); ++i){
      PlayerMove pm = mHistory.back();
      mHistory.pop_back();
      if (pm.move.mty != M::Play) continue;

      if (pm.move.mty == M::Play){
        mBoard.revert(pm.move.mpt);
      }
    }
    if (mHistory.size() > 0) mNPlayer = opponent(mHistory.back().player);
    else                     mNPlayer = Player::Black;
    run_test_termination();
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
  // this function need to be efficient
  void run_test_termination(){
    auto [is_terminal, winner] = compute_termination<R,C,X>(mBoard);
    mTerminal = is_terminal;
    mWinner = winner;
  }
};

} // rlbg

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
} // std

#endif//RLBG_CONNECTX
