#include <sstream>
#include <string>
#include <unordered_set>

#include <gtest/gtest.h>

#include <types.h>

namespace r = rlbg;

namespace {

template <typename T>
std::string str(const T& v){
  std::ostringstream os;
  os << v;
  return os.str();
}

} // namespace

TEST(Types, PointArithmetic){
  r::Pt p(3, 4);
  EXPECT_EQ(p + r::Dr(1, -1), r::Pt(4, 3));
  EXPECT_EQ(r::Dr(1, -1) + p, r::Pt(4, 3));
  EXPECT_EQ(p - r::Dr(1, -1), r::Pt(2, 5));
  p += r::Dr(-3, 2);
  EXPECT_EQ(p, r::Pt(0, 6));
  p -= r::Dr(0, 6);
  EXPECT_EQ(p, r::Pt(0, 0));
  EXPECT_NE(r::Pt(1, 2), r::Pt(2, 1));
}

TEST(Types, IndexPointRoundTrip){
  for (uint i = 0; i < 64; ++i)
    EXPECT_EQ(r::index<8>(r::point<8>(i)), i);
  EXPECT_EQ(r::index<7>(r::Pt(2, 3)), 17u);
}

TEST(Types, Neighbourhoods){
  r::Pt p(4, 4);
  for (r::Pt q : r::neighbours(p)) EXPECT_EQ(std::abs(q.r - p.r) + std::abs(q.c - p.c), 1);
  for (r::Pt q : r::corners(p)) EXPECT_TRUE(std::abs(q.r - p.r) == 1 and std::abs(q.c - p.c) == 1);
  std::unordered_set<r::Pt> around;
  for (r::Pt q : r::surroundings(p)) around.insert(q);
  EXPECT_EQ(around.size(), 8u);
  EXPECT_FALSE(around.contains(p));
}

TEST(Types, PlayersAndMoves){
  EXPECT_EQ(r::opponent(r::Player::Black), r::Player::White);
  EXPECT_EQ(r::opponent(r::Player::White), r::Player::Black);
  // pass and resign compare by type only
  EXPECT_EQ(r::Move(r::M::Pass), r::Move(r::M::Pass, r::Pt(3, 3)));
  EXPECT_NE(r::Move(r::M::Play, r::Pt(1, 1)), r::Move(r::M::Play, r::Pt(1, 2)));
  EXPECT_NE(r::Move(r::M::Pass), r::Move(r::M::Resign));
  EXPECT_EQ(r::PlayerMove(r::Player::Black, r::Move(r::M::Pass)), r::PlayerMove(r::Player::Black, r::Move(r::M::Pass)));
  EXPECT_NE(r::PlayerMove(r::Player::Black, r::Move(r::M::Pass)), r::PlayerMove(r::Player::White, r::Move(r::M::Pass)));
}

TEST(Types, Printing){
  EXPECT_EQ(str(r::Pt(2, 5)), "(2,5)");
  EXPECT_EQ(str(r::Move(r::M::Play, r::Pt(0, 1))), "(0,1)");
  EXPECT_EQ(str(r::Move(r::M::Pass)), "Pass");
  EXPECT_EQ(str(r::Player::Unknown), "Tie");
}
