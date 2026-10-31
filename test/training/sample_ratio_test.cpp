#include <algorithm>

#include <gtest/gtest.h>

#include <training/sample_ratio.h>

namespace r = rlbg;

TEST(SampleRatio, LearnerWaitsForTheFirstPositions){
  r::SampleRatio ratio(256, 2., 1000, 500);
  EXPECT_FALSE(ratio.learner_may_step(0, 5000, 999));
  EXPECT_TRUE(ratio.learner_may_step(0, 5000, 1000));
}

TEST(SampleRatio, LearnerKeepsTheReuseRatio){
  r::SampleRatio ratio(100, 2., 0, 0);
  // step k + 1 draws (k + 1) * 100 samples, allowed up to 2 per position
  EXPECT_TRUE(ratio.learner_may_step(9, 500, 500));     // 1000 <= 1000
  EXPECT_FALSE(ratio.learner_may_step(10, 500, 500));   // 1100 > 1000
  EXPECT_TRUE(ratio.learner_may_step(10, 550, 550));
}

TEST(SampleRatio, SelfPlayMayAlwaysFillTheMinimum){
  r::SampleRatio ratio(256, 2., 1000, 500);
  EXPECT_FALSE(ratio.selfplay_should_pause(0, 1500, 1000));
  EXPECT_TRUE(ratio.selfplay_should_pause(0, 1501, 1000));
}

// regression: with many games in flight, far more moves are played than
// finished positions reach the replay buffer; self-play must keep going
// until the learner has data
TEST(SampleRatio, SelfPlayRunsUntilFinishedGamesFillTheMinimum){
  r::SampleRatio ratio(16, 4., 200, 200);
  EXPECT_FALSE(ratio.selfplay_should_pause(0, 100000, 0));
  EXPECT_FALSE(ratio.selfplay_should_pause(0, 100000, 199));
  EXPECT_TRUE(ratio.selfplay_should_pause(0, 100000, 200));
}

TEST(SampleRatio, SelfPlayWaitsWhenTooFarAhead){
  r::SampleRatio ratio(100, 2., 0, 50);
  // after 10 steps (1000 samples) the learner wants 500 positions
  EXPECT_FALSE(ratio.selfplay_should_pause(10, 550, 550));
  EXPECT_TRUE(ratio.selfplay_should_pause(10, 551, 551));
  EXPECT_FALSE(ratio.selfplay_should_pause(20, 1000, 1000));
}

// finished positions trail the moves played by anything up to all the moves
// of games in progress; no split may stop both sides
TEST(SampleRatio, BothSidesNeverWaitAtOnce){
  r::SampleRatio ratio(64, 3., 200, 100);
  for (int step = 0; step < 100; ++step)
    for (double played = 0; played < 3000; played += 7)
      for (double in_flight : {0., 50., 500., 5000.}){
        size_t finished = (size_t)std::max(0., played - in_flight);
        bool learner_waits = not ratio.learner_may_step(step, played, finished);
        bool selfplay_waits = ratio.selfplay_should_pause(step, played, finished);
        EXPECT_FALSE(learner_waits and selfplay_waits) << "deadlock at step " << step << " played " << played
                                                        << " in flight " << in_flight;
      }
}

TEST(SampleRatio, TinyLeadsCannotDeadlock){
  // a lead below one batch worth of positions would let both sides wait
  r::SampleRatio ratio(256, 2., 0, 0);
  for (int step = 1; step < 50; ++step)
    for (double played = 0; played < 8000; played += 3)
      EXPECT_FALSE(not ratio.learner_may_step(step, played, (size_t)played) and
                   ratio.selfplay_should_pause(step, played, (size_t)played)) << step << " " << played;
}
