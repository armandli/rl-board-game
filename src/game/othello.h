#ifndef RLBG_OTHELLO
#define RLBG_OTHELLO

#include <cassert>
#include <cstdint>

#include <array>
#include <bit>
#include <ostream>
#include <vector>

#include <type_alias.h>
#include <types.h>
#include <zobrist_hash.h>
#include <game_base.h>

namespace rlbg {

namespace s = std;

// bitboard with bit index r * SZ + c; the flip stack keeps the discs flipped
// by each placement so revert can undo it
template <ubyte SZ>
struct OthelloBoard : Board<OthelloBoard<SZ>> {
  static constexpr uint IZ = SZ * SZ;
  static_assert(IZ <= 64, "Othello Board must fit in a 64 bit bitboard");

  OthelloBoard(): mBlack(0), mWhite(0) {
    static_assert(SZ % 2 == 0, "Othello Board size must be divisible by 2");

    mBlack = bit(Pt(SZ/2-1,SZ/2))   | bit(Pt(SZ/2,  SZ/2-1));
    mWhite = bit(Pt(SZ/2-1,SZ/2-1)) | bit(Pt(SZ/2  ,SZ/2));
  }
  OthelloBoard(const OthelloBoard& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mFlipStack = o.mFlipStack;
  }
  OthelloBoard& operator=(const OthelloBoard& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mFlipStack = o.mFlipStack;
    return *this;
  }
  OthelloBoard(OthelloBoard&& o) noexcept
  : mBlack(o.mBlack),
    mWhite(o.mWhite),
    mFlipStack(s::move(o.mFlipStack))
  {}
  OthelloBoard& operator=(OthelloBoard&& o){
    mBlack = o.mBlack;
    mWhite = o.mWhite;
    mFlipStack = s::move(o.mFlipStack);
    return *this;
  }

  // placements for player as a bitmask
  uint64_t legal_mask(Player player) const {
    return legal(pieces(player), pieces(opponent(player)));
  }
  uint64_t pieces(Player player) const {
    switch (player){
    case Player::Black:
      return mBlack;
    break; case Player::White:
      return mWhite;
    break; case Player::Unknown: default:
      assert(false);
    }
    return 0;
  }
  size_t count_player_piece(Player player) const {
    return s::popcount(pieces(player));
  }

  bool is_valid(Player player, Pt pt) const {
    if (not is_on_board(pt)) return false;
    return legal_mask(player) & bit(pt);
  }
  Player operator[](Pt pt) const {
    assert(pt.r >= 0 and pt.r < SZ and pt.c >= 0 and pt.c < SZ);
    uint64_t m = bit(pt);
    if (mBlack & m) return Player::Black;
    if (mWhite & m) return Player::White;
    return Player::Unknown;
  }
  void place(Player player, Pt pt){
    assert(is_valid(player, pt));
    uint64_t& own = player == Player::Black ? mBlack : mWhite;
    uint64_t& opp = player == Player::Black ? mWhite : mBlack;
    uint64_t f = flips(own, opp, bit(pt));
    own |= f | bit(pt);
    opp &= ~f;
    mFlipStack.push_back(f);
  }
  void revert(Pt pt){
    Player rp = (*this)[pt];
    assert(rp != Player::Unknown);
    uint64_t& own = rp == Player::Black ? mBlack : mWhite;
    uint64_t& opp = rp == Player::Black ? mWhite : mBlack;
    own &= ~bit(pt);

    if (not mFlipStack.empty()){
      own &= ~mFlipStack.back();
      opp |= mFlipStack.back();
      mFlipStack.pop_back();
    }
  }
  size_t size() const {
    return IZ;
  }
  size_t hash() const {
    return mix64(mBlack) ^ mix64(mWhite ^ 0x9e3779b97f4a7c15ULL);
  }
  s::ostream& print(s::ostream& out) const {
    char bchar = 'X';
    char wchar = '0';

    out << "   ";
    for (uint c = 0; c < SZ; ++c){
      out << c;
    }
    out << '\n';

    for (uint r = 0; r < SZ; ++r){
      out << ' ' << r << ' ';
      for (uint c = 0; c < SZ; ++c){
        switch ((*this)[Pt(r,c)]){
        case Player::Black:
          out << bchar;
        break; case Player::White:
          out << wchar;
        break; case Player::Unknown:
          out << ' ';
        break; default: assert(false);
        }
      }
      out << ' ' << r << '\n';
    }
    return out;
  }

protected:
  static constexpr uint64_t BOARD = IZ == 64 ? ~0ULL : (1ULL << IZ) - 1;
  static constexpr uint64_t column_mask(uint c){
    uint64_t m = 0;
    for (uint r = 0; r < SZ; ++r)
      m |= 1ULL << (r * SZ + c);
    return m;
  }
  // runs that move across columns must not wrap around the board edge, so
  // they can only pass through the inner columns
  static constexpr uint64_t INNER = BOARD & ~column_mask(0) & ~column_mask(SZ-1);
  // east, south, south-east, south-west
  static constexpr int      DIRS[4]  = {1, SZ, SZ + 1, SZ - 1};
  static constexpr uint64_t MASKS[4] = {INNER, BOARD, INNER, INNER};

  static bool is_on_board(Pt pt){
    return pt.r >= 0 and pt.r < SZ and pt.c >= 0 and pt.c < SZ;
  }
  static uint64_t bit(Pt pt){
    return 1ULL << index<SZ>(pt);
  }
  static uint64_t shift(uint64_t b, int d){
    return d > 0 ? b << d : b >> -d;
  }
  // squares one step beyond each run of mo that starts next to p
  static uint64_t sweep(uint64_t p, uint64_t mo, int d){
    uint64_t x = shift(p, d) & mo;
    for (int i = 0; i < (int)SZ - 3; ++i)
      x |= shift(x, d) & mo;
    return shift(x, d);
  }
  // placements for p against o, flood fill along the eight directions
  static uint64_t legal(uint64_t p, uint64_t o){
    uint64_t empty = ~(p | o) & BOARD;
    uint64_t moves = 0;
    for (int i = 0; i < 4; ++i){
      uint64_t mo = o & MASKS[i];
      moves |= sweep(p, mo, DIRS[i]) & empty;
      moves |= sweep(p, mo, -DIRS[i]) & empty;
    }
    return moves;
  }
  // discs of o flipped by p placing on square m
  static uint64_t flips(uint64_t p, uint64_t o, uint64_t m){
    uint64_t f = 0;
    for (int i = 0; i < 4; ++i){
      uint64_t mo = o & MASKS[i];
      for (int d : {DIRS[i], -DIRS[i]}){
        uint64_t x = 0;
        uint64_t t = shift(m, d);
        while (t & mo){
          x |= t;
          t = shift(t, d);
        }
        if (t & p) f |= x;
      }
    }
    return f;
  }

  uint64_t            mBlack;
  uint64_t            mWhite;
  s::vector<uint64_t> mFlipStack;
};

template <ubyte SZ>
s::ostream& operator<<(s::ostream& out, const OthelloBoard<SZ>& board){
  return board.print(out);
}

template <ubyte SZ>
struct OthelloGameState : GameState<OthelloBoard<SZ>, OthelloGameState<SZ>> {
  OthelloGameState()
  : mBoard(), mNPlayer(Player::Black), mHistory(),
    mWinner(Player::Unknown), mTerminal(false)
  {
    mLegal = mBoard.legal_mask(mNPlayer);
  }
  OthelloGameState(
      const OthelloBoard<SZ>& board,
      Player nxt_player,
      const s::vector<PlayerMove>& history,
      bool term,
      Player winner)
  : mBoard(board), mNPlayer(nxt_player), mHistory(history),
    mWinner(winner), mTerminal(term)
  {
    mLegal = mBoard.legal_mask(mNPlayer);
  }
  OthelloGameState(const OthelloGameState& o)
  : mBoard(o.mBoard),
    mNPlayer(o.mNPlayer),
    mHistory(o.mHistory),
    mLegal(o.mLegal),
    mWinner(o.mWinner),
    mTerminal(o.mTerminal)
  {}
  OthelloGameState& operator=(const OthelloGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mHistory = o.mHistory;
    mLegal = o.mLegal;
    mWinner = o.mWinner;
    mTerminal = o.mTerminal;
    return *this;
  }
  OthelloGameState(OthelloGameState&& o)
  : mBoard(s::move(o.mBoard)),
    mNPlayer(o.mNPlayer),
    mHistory(s::move(o.mHistory)),
    mLegal(o.mLegal),
    mWinner(o.mWinner),
    mTerminal(o.mTerminal)
  {}
  OthelloGameState& operator=(OthelloGameState&& o){
    mBoard    = s::move(o.mBoard);
    mNPlayer  = o.mNPlayer;
    mHistory  = s::move(o.mHistory);
    mLegal    = o.mLegal;
    mWinner   = o.mWinner;
    mTerminal = o.mTerminal;
    return *this;
  }

  const OthelloBoard<SZ>& board() const {
    return mBoard;
  }
  Player next_player() const {
    return mNPlayer;
  }
  bool is_valid(PlayerMove pm) const {
    if (is_terminal()) return false;
    switch (pm.move.mty){
    case M::Play:
      if (pm.player == mNPlayer)
        return is_on_board(pm.move.mpt) and (mLegal >> index<SZ>(pm.move.mpt)) & 1ULL;
      return mBoard.is_valid(pm.player, pm.move.mpt);
    break; case M::Pass: case M::Resign:
      return true;
    break; case M::Unknown:
      return false;
    break; default: assert(false);
    }
    return false;
  }
  MoveList<SZ * SZ + 1> legal_moves() const {
    MoveList<SZ * SZ + 1> moves;
    if (is_terminal()) return moves;

    if (mHistory.size() > 0){
      if (mHistory.back().move.mty == M::Pass)
        moves.push_back(Move(M::Resign));
      else
        moves.push_back(Move(M::Pass));
    } else
      moves.push_back(Move(M::Pass));
    for (uint64_t m = mLegal; m; m &= m - 1)
      moves.push_back(Move(M::Play, point<SZ>(s::countr_zero(m))));
    return moves;
  }
  bool apply(PlayerMove pm){
    if (is_terminal())    return false;
    if (not is_valid(pm)) return false;
    if (pm.player != mNPlayer) return false;
    switch (pm.move.mty){
    case M::Play:
      mBoard.place(pm.player, pm.move.mpt);
      mNPlayer = opponent(mNPlayer);
      mHistory.push_back(pm);
      update_termination();
      return true;
    break; case M::Pass:
      mNPlayer = opponent(mNPlayer);
      mHistory.push_back(pm);
      update_termination();
      return true;
    break; case M::Resign:
      mTerminal = true;
      mWinner = opponent(pm.player);
      mHistory.push_back(pm);
      mNPlayer = opponent(mNPlayer);
      mLegal = mBoard.legal_mask(mNPlayer);
      return true;
    break; case M::Unknown: default:
      return false;
    }
  }
  void rollback(size_t steps){
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
    mLegal = mBoard.legal_mask(mNPlayer);
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
    if (mHistory.size() > 0){
      if (mHistory.back().move.mty == M::Pass){
        h ^= PASS_HASH;
      } else if (mHistory.back().move.mty == M::Resign){
        h ^= RESIGN_HASH;
      }
    }
    return h;
  }
  const s::vector<PlayerMove>& history() const {
    return mHistory;
  }

protected:
  static bool is_on_board(Pt pt){
    return pt.r >= 0 and pt.r < SZ and pt.c >= 0 and pt.c < SZ;
  }
  // terminal when neither side can place; only checks the opponent when the
  // side to move is stuck
  void update_termination(){
    mLegal = mBoard.legal_mask(mNPlayer);
    mTerminal = mLegal == 0 and mBoard.legal_mask(opponent(mNPlayer)) == 0;
    if (mTerminal){
      int black_count = mBoard.count_player_piece(Player::Black);
      int white_count = mBoard.count_player_piece(Player::White);
      switch ((black_count>white_count) - (white_count>black_count)){
      case 1:
        mWinner = Player::Black;
      break; case 0:
        mWinner = Player::Unknown;
      break; case -1:
        mWinner = Player::White;
      break; default: assert(false); // should never get here
      }
    }
  }

  OthelloBoard<SZ>      mBoard;
  Player                mNPlayer;
  s::vector<PlayerMove> mHistory;
  // placements for mNPlayer
  uint64_t              mLegal;
  Player                mWinner;
  bool                  mTerminal;
};

}  // namespace rlbg

namespace r = rlbg;

namespace std {
template <ubyte SZ>
struct hash<r::OthelloBoard<SZ>> {
  size_t operator()(const r::OthelloBoard<SZ>& board) const {
    return board.hash();
  }
};
template <ubyte SZ>
struct hash<r::OthelloGameState<SZ>> {
  size_t operator()(const r::OthelloGameState<SZ>& state) const {
    return state.hash();
  }
};
}  // namespace std

#endif//RLBG_OTHELLO
