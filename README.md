# rl-board-game
RL board game environment

## Model based reinforcement learning (MuZero)

`othello_mcgs_selfplay_train` trains an othello player with MuZero: a learned
model (representation, dynamics, prediction) searched by Monte-Carlo graph
search, trained on its own self-play games. It needs MLX.

```sh
make
build/bin/othello_mcgs_selfplay_train                 # train, resuming from data/
build/bin/othello_mcgs_selfplay_train --print-config  # every option with its default
build/bin/othello_mcgs_selfplay_train --config my.toml --simulations 128
```

Ctrl-C stops after saving a checkpoint; the next run resumes from it.
Everything is written under `--data-dir` (default `data/`):

| path | contents |
|------|----------|
| `experience/games_*.bin` | every self-play game; reloaded into the replay buffer on resume |
| `checkpoint/` | model, optimizer state, step and data counters |
| `snapshots/step_*.safetensors` | model snapshots, one every `--snapshot-interval` steps |
| `eval/results.csv`, `eval/ratings.csv` | snapshot matches and Elo ratings (random player = 0) |
| `train_log.csv` | losses and throughput over time |
| `config.toml` | the options of the last run |

Progress shows up as `[eval]` lines: every snapshot plays older snapshots,
the random player and a UCT random playout baseline, and all results are
refit into Elo ratings. Rising Elo, and a growing score against the fixed
`mcts_rollout` baseline, mean the model keeps improving.

### Components

Each component is an interface with a first implementation that can be
swapped out:

| directory | interface | implementation |
|-----------|-----------|----------------|
| `game/` | `MuZeroGame` concept | `OthelloBitboard`: bitboard othello for fast search |
| `encoder/` | `Encoder<Game>`: state to token features, board symmetries | `OthelloEncoder`: a token per square, side-to-move relative, 8 symmetries |
| `model/` | `MuZeroModel`: represent, dynamics, predict + auxiliary heads | `TransformerMuZero`: pre-norm transformer (RMSNorm, SwiGLU, fused attention) over board tokens |
| `search/` | `Search<Game>`, `Inference` | `MCGS`: Monte-Carlo graph search on the learned model with a Gumbel MuZero or PUCT root |
| `selfplay/` | | `SelfPlayActor`, `ReanalyzeActor`, `InferenceServer` batching all actors on the GPU |
| `experience/` | `ReplayBuffer`, `ExperienceStore` | `FifoReplayBuffer` (uniform over recent positions), `BinaryExperienceStore` |
| `training/` | `Learner` | `MuZeroLearner`, `BatchBuilder`, `TrainingLoop` running everything in parallel |
| `evaluation/` | `Agent<Game>`, `Evaluator` | `NetworkAgent`, `RolloutMCTSAgent`, `RandomAgent`, `SnapshotEvaluator` with Elo |

Techniques, by source:

- MuZero (Schrittwieser et al. 2020): learned latent model, K = 5 step unroll
  training, gradient halving into the dynamics, Reanalyse (MuZero Unplugged)
- EfficientZero (Ye et al. 2021): SimSiam consistency loss between the unrolled
  latent and the encoded real next state
- Scholz et al. 2021: reconstruction of the board from every latent
- Gumbel MuZero (Danihelka et al. 2022): policy improvement that works with
  few simulations
- Monte-Carlo graph search (Czech et al. 2020; KataGo's formulation): the real
  rules detect transpositions so each position is evaluated once
- KataGo (Wu 2019): playout cap randomization, final ownership auxiliary target
- Muon optimizer for the transformer's weight matrices, bfloat16 training,
  float16 inference

The learner and self-play share the GPU at a fixed ratio
(`--sample-reuse` training samples per self-play position): whichever side
gets ahead waits, so the same settings balance any machine. Thread counts
default to the number of cores.

## Tests

```sh
make test                                  # build, then run every suite with CTest
ctest --test-dir build -L search_test      # one suite
build/bin/training_test --gtest_filter='SampleRatio*'
```

GoogleTest suites live in `test/`, one executable per component:

| suite | covers |
|-------|--------|
| `game_test` | geometry types, `OthelloGameState`, `OthelloBitboard` (checked against `OthelloGameState`), `ConnectXGameState` (checked against a brute force line counter), minimax and rollout graph search |
| `encoder_test` | features, token classes, symmetries form D4 and preserve the rules |
| `experience_test` | replay buffer windowing, uniform sampling, thread safety; experience file round trip, rotation, truncation |
| `mlx_util_test` | modules, io, gradients, optimizers, transformer blocks |
| `model_test` | `TransformerMuZero` shapes, normalization, action conditioning, weights io; `ModelInference` batching, precision, threads |
| `search_test` | Gumbel schedule; MCGS against exact models of tic-tac-toe and othello: tactics, transpositions, determinism, strength |
| `selfplay_test` | trajectory recording, inference server batching, self-play and reanalyze actors |
| `evaluation_test` | Elo fit, arena, agents, snapshot evaluator with restart |
| `training_test` | sample ratio control, batch targets and masks under symmetries, learner losses and checkpoints, a full training run and its resume |

`test/common/` holds the shared fixtures: tic-tac-toe as a second
`MuZeroGame`, "true models" that use the real rules as the dynamics, small
transformer configurations and random trajectories.
