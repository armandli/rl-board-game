#ifndef RLBG_ENCODER_ENCODER_H
#define RLBG_ENCODER_ENCODER_H

#include <cstdint>
#include <vector>

namespace rlbg {

namespace s = std;

// turns a game state into the network input: a sequence of num_tokens()
// tokens (board squares) each with num_features() small integer features,
// stored row major as uint8 so trajectories stay compact in the replay buffer.
// Every token also has a class in [0, num_token_classes()) seen from the side
// to move, used as the target of the observation reconstruction and final
// ownership heads.
//
// Symmetries are permutations of tokens and actions under which the rules are
// invariant; they augment training data. token_map(sym)[i] is where token i
// moves to, action_map(sym)[a] is where action a moves to.
template <typename Game>
struct Encoder {
  virtual ~Encoder() = default;

  virtual int num_tokens() const = 0;
  virtual int num_features() const = 0;
  virtual int num_actions() const = 0;
  virtual int num_token_classes() const = 0;
  int observation_size() const { return num_tokens() * num_features(); }

  // writes observation_size() values
  virtual void encode(const Game& game, uint8_t* features) const = 0;
  // writes num_tokens() classes seen from the side to move
  virtual void encode_tokens(const Game& game, uint8_t* tokens) const = 0;
  // writes num_tokens() classes seen from player 0
  virtual void encode_tokens_absolute(const Game& game, uint8_t* tokens) const = 0;
  // the class of a token seen from the other player
  virtual int flip_token_class(int token_class) const = 0;

  virtual int num_symmetries() const { return 1; }
  virtual const s::vector<int>& token_map(int sym) const = 0;
  virtual const s::vector<int>& action_map(int sym) const = 0;
};

}  // namespace rlbg

#endif//RLBG_ENCODER_ENCODER_H
