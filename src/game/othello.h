#ifndef RLBG_OTHELLO
#define RLBG_OTHELLO

#include <cassert>
#include <array>
#include <unordered_set>

#include <type_alias.h>
#include <types.h>
#include <game_base.h>

namespace rlbg {

namespace s = std;

template <ubyte SZ>
struct OthelloBoard : Board<OthelloBoard<SZ>> {
  static constexpr uint IZ = SZ * SZ;
protected:
  s::array<Player, IZ> mBoard;
  s::unordered_set<Pt> mBlackCache;
  s::unordered_set<Pt> mWhiteCache;
public:
  OthelloBoard(){
    s::memset(mBoard.data(), (uint)Player::Unknown, sizeof(ubyte)*IZ);
    mBoard[index<SZ>(Pt(SZ/2-1,SZ/2))]   = Player::Black;
    mBoard[index<SZ>(Pt(SZ/2,  SZ/2-1))] = Player::Black;
    mBoard[index<SZ>(Pt(SZ/2-1,SZ/2-1))] = Player::White;
    mBoard[index<SZ>(Pt(SZ/2  ,SZ/2))]   = Player::White;

    for (Pt pt : surroundings(Pt(SZ/2-1,SZ/2)))
      if (check_placement_valid(Player::White, pt))
        mWhiteCache.insert(pt);
    for (Pt pt : surroundings(Pt(SZ/2,SZ/2-1)))
      if (check_placement_valid(Player::White, pt))
        mWhiteCache.insert(pt);
    for (Pt pt : surroundings(Pt(SZ/2-1,SZ/2-1)))
      if (check_placement_valid(Player::Black, pt))
        mBlackCache.insert(pt);
    for (Pt pt : surroundings(Pt(SZ/2,SZ/2)))
      if (check_placement_valid(Player::Black, pt))
        mBlackCache.insert(pt);
  }
  OthelloBoard(const OthelloBoard& o){
    mBoard = o.mBoard;
    mBlackCache = o.mBlackCache;
    mWhiteCache = o.mWhiteCache;
  }
  OthelloBoard& operator=(const OthelloBoard& o){
    mBoard = o.mBoard;
    mBlackCache = o.mBlackCache;
    mWhiteCache = o.mWhiteCache;
    return *this;
  }
  OthelloBoard(OthelloBoard&& o) noexcept :
    mBoard(s::move(o.mBoard)),
    mBlackCache(s::move(o.mBlackCache)),
    mWhiteCache(s::move(o.mWhiteCache))
  {}
  OthelloBoard& operator=(OthelloBoard&& o){
    //TODO: correct ?
    mBoard = s::move(o.mBoard);
    mBlackCache = s::move(o.mBlackCache);
    mWhiteCache = s::move(o.mWhiteCache);
    return *this;
  }

  const s::unordered_set<Pt>& get_legal_moves(Player player) const {
    switch (player){
    break; case Player::Black:
      return mBlackCache;
    break; case Player::White:
      return mWhiteCache;
    break; case Player::Unknown:
           default:
      assert(false);
    }
  }
  size_t count_player_piece(Player player) const {
    size_t sum = 0;
    for (uint i = 0; i < IZ; ++i)
      if (mBoard[i] == player)
        sum += 1;
    return sum;
  }

  bool is_valid(Player player, Pt pt) const {
    if (not is_on_board(pt)) return false;
    if (mBoard[index<SZ>(pt)] != Player::Unknown) return false;

    switch (player){
    break; case Player::Black:
      return mBlackCache.contains(pt);
    break; case Player::White:
      return mWhiteCache.contains(pt);
    break; case Player::Unknown:
           default:
      assert(false);
    }
  }
  Player operator[](Pt pt) const {
    assert(pt.r < SZ && pt.c < SZ);
    return mBoard[index<SZ>(pt)];
  }
  void place(Player player, Pt pt){
    assert(is_valid(player, pt));
    // scan 8 directions, and flip opponent pieces if caught in between
    mBoard[index<SZ>(pt)] = player;
    for (Direction d : EIGHT_WAY){
      Pt bp = pt + d;
      flip_stack(player, bp, d);
    }
    // scan 8 directions, and flip opponent pieces if caught in between
    // remove moves in cache no longer valid
    s::unordered_set<Pt> to_remove;
    for (Pt p : mBlackCache)
      if (not check_placement_valid(Player::Black, p))
        to_remove.insert(p);
    for (Pt p : to_remove)
      mBlackCache.erase(p);
    to_remove.clear();
    for (Pt p : mWhiteCache)
      if (not check_placement_valid(Player::White, p))
        to_remove.insert(p);
    for (Pt p : to_remove)
      mWhiteCache.erase(p);
    //TODO: we could probably do better by keeping track of locations where
    //pieces has been flipped, and check for those locations only
    // naive approach to add new possible moves
    for (uint i = 0; i < IZ; ++i){
      if (mBoard[i] != Player::Unknown){
        for (Pt p : surroundings(point<SZ>(i))){
          if (check_placement_valid(Player::Black, p))
            mBlackCache.insert(p);
          if (check_placement_valid(Player::White, p))
            mWhiteCache.insert(p);
        }
      }
    }
  }
  size_t size() const {
    return IZ;
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
        switch (mBoard[index<SZ>(Pt(r,c))]){
        break; case Player::Black:
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
  bool is_on_board(Pt pt) const {
    return pt.r < SZ && pt.c < SZ;
  }
  bool check_placement_valid(Player player, Pt pt) const {
    if (not is_on_board(pt)) return false;
    if (mBoard[index<SZ>(pt)] != Player::Unknown) return false;

    for (Direction d : EIGHT_WAY){
      bool found_opponent = false;
      for (Pt p = pt + d; is_on_board(p); p += d){
        if (mBoard[index<SZ>(p)] == Player::Unknown)
          break;
        if (not found_opponent && mBoard[index<SZ>(p)] == player)
          break;
        if (not found_opponent && mBoard[index<SZ>(p)] != player){
          found_opponent = true;
          continue;
        }
        if (found_opponent && mBoard[index<SZ>(p)] == player)
          return true;
      }
    }
    return false;
  }
  Player flip_stack(Player player, Pt pt, Direction d){
    if (not is_on_board(pt)) return opponent(player);
    if (mBoard[index<SZ>(pt)] == Player::Unknown) return opponent(player);
    if (mBoard[index<SZ>(pt)] == player) return player;

    Player p = flip_stack(player, pt+d, d);
    mBoard[index<SZ>(pt)] = p;
    return p;
  }
};

template <ubyte SZ>
s::ostream& operator<<(s::ostream& out, const OthelloBoard<SZ>& board){
  return board.print(out);
}

template <ubyte SZ>
struct OthelloGameState : GameState<OthelloBoard<SZ>, OthelloGameState<SZ>> {
protected:
  OthelloBoard<SZ>      mBoard;
  Player                mNPlayer;
  s::vector<PlayerMove> mHistory;
  Player                mWinner;
  bool                  mTerminal;
public:
  OthelloGameState(): mBoard(), mNPlayer(Player::Black), mHistory(), mWinner(Player::Unknown), mTerminal(false) {}
  OthelloGameState(const OthelloBoard<SZ>& board, Player nxt_player, const s::vector<PlayerMove>& history, bool term, Player winner): mBoard(board), mNPlayer(nxt_player), mHistory(history), mWinner(winner), mTerminal(term) {}
  OthelloGameState(const OthelloGameState& o): mBoard(o.mBoard), mNPlayer(o.mNPlayer), mHistory(o.mHistory), mWinner(o.mWinner), mTerminal(o.mTerminal) {}
  OthelloGameState& operator=(const OthelloGameState& o){
    mBoard = o.mBoard;
    mNPlayer = o.mNPlayer;
    mHistory = o.mHistory;
    mWinner = o.mWinner;
    mTerminal = o.mTerminal;
    return *this;
  }
  OthelloGameState(OthelloGameState&& o): mBoard(s::move(o.mBoard)), mNPlayer(o.mNPlayer), mHistory(s::move(o.mHistory)), mWinner(o.mWinner), mTerminal(o.mTerminal) {}
  OthelloGameState& operator=(OthelloGameState&& o){
    //TODO: correct ?
    mBoard   = s::move(o.mBoard);
    mNPlayer = o.mNPlayer;
    mHistory = s::move(o.mHistory);
    mWinner  = o.mWinner;
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
    break; case M::Play:
      return mBoard.is_valid(pm.player, pm.move.mpt);
    break; case M::Pass:
      return true;
    break; case M::Resign:
      return true;
    break; case M::Unknown:
      return false;
    break; default: assert(false);
    }
  }
  s::vector<Move> legal_moves() const {
    if (is_terminal()) return s::vector<Move>();
    s::vector<Move> moves;
    moves.push_back(Move(M::Pass));
    moves.push_back(Move(M::Resign));
    const s::unordered_set<Pt>& pts = mBoard.get_legal_moves(mNPlayer);
    for (Pt pt : pts)
      moves.push_back(Move(M::Play, pt));
    return moves;
  }
  OthelloGameState& apply(PlayerMove pm){
    if (is_terminal())    return *this;
    if (not is_valid(pm)) return *this;
    if (pm.player != mNPlayer) return *this;
    switch (pm.move.mty){
    break; case M::Play:
      mBoard.place(pm.player, pm.move.mpt);
      mNPlayer = opponent(mNPlayer);
      mHistory.push_back(pm);
      mTerminal = check_no_more_moves();
      if (mTerminal){
        int black_count = mBoard.count_player_piece(Player::Black);
        int white_count = mBoard.count_player_piece(Player::White);
        switch ((black_count>white_count) - (white_count>black_count)){
        break; case 1:
          mWinner = Player::Black;
        break; case 0:
          mWinner = Player::Unknown;
        break; case -1:
          mWinner = Player::White;
        }
      }
      return *this;
    break; case M::Pass:
      mNPlayer = opponent(mNPlayer);
      mHistory.push_back(pm);
      return *this;
    break; case M::Resign:
      mTerminal = true;
      mWinner = opponent(pm.player);
      mHistory.push_back(pm);
      return *this;
    break; case M::Unknown:
           default:
      return *this;
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
  const s::vector<PlayerMove>& history() const {
    return mHistory;
  }
protected:
  bool check_no_more_moves() const {
    return mBoard.get_legal_moves(Player::Black).size() +
           mBoard.get_legal_moves(Player::White).size() == 0;
  }
};

} // rlbg

#endif//RLBG_OTHELLO
