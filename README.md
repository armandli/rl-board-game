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

## MuZero on LibTorch (CUDA)

The same pipeline with a LibTorch model, sized for one consumer GPU (tuned on
an RTX 5080 laptop GPU, 16 GB, with a 24 thread CPU). The search, self-play,
replay, batching and evaluation code is shared with the MLX version; the
framework sits behind `TrainingBackend` (`src/training/training_backend.h`).

```sh
make
build/bin/othello_muzero_torch_train                       # train, resuming from data/othello_muzero_torch
build/bin/othello_muzero_torch_train --max-minutes 480     # stop after 8 hours (checkpoint + final model)
build/bin/othello_muzero_torch_train --print-config        # every option with its default
build/bin/othello_muzero_torch_play                        # play the final model
build/bin/othello_muzero_torch_play -m data/othello_muzero_torch/snapshots/step_00020000.pt -s 800 -c white
```

Ctrl-C stops after saving a checkpoint, and the next run resumes from it.
Along with the files listed above, a run writes `snapshots/step_*.pt` and
`final_model.pt` (the weights at the end of the last run).

### Playing

`othello_muzero_torch_play` is a full screen terminal app. Click a square to
place a disc (arrow keys + Enter work too), `u` undoes your last move, `n`
starts a new game, `s` swaps colors, `h` toggles legal move hints, `q` quits.
The model searches on a background thread with `--simulations` (default
256) and shows its winning chances after each move. Passes are automatic,
because othello only allows passing when there is no legal placement.

### Checking that training improves

Every snapshot is rated while training (`[eval]` lines, `eval/ratings.csv`)
against older snapshots, the random player, `mcts_rollout_1000` (UCT with
random playouts) and `alphabeta_4`, a classic engine (negamax alpha-beta,
square table + mobility + corners, exact in the last 8 empties). Two fixed
baselines that know nothing learned make the ratings meaningful across runs.
For a less noisy check, rate snapshots on a fixed yardstick:

```sh
build/bin/othello_muzero_torch_arena --snapshots data/othello_muzero_torch/snapshots --every 5 \
  --opponent alphabeta:2,alphabeta:4,rollout:1000 --games 200 --csv progress.csv
build/bin/othello_muzero_torch_arena --model new.pt --opponent model:old.pt --games 400
```

The arena plays color-swapped pairs from the same random openings for every
model, so scores compare across snapshots and runs.

### Model

`TransformerMuZero` (`src/libtorch/`) uses a game state embedding of 64 board
tokens. Each token is a learned embedding of the square's class (empty, own,
opponent; side-to-move relative), plus a projection of its other features
(legal placement), plus a learned position embedding. Representation,
dynamics and prediction are pre-norm transformers (fused RMSNorm, SwiGLU,
`scaled_dot_product_attention`). The dynamics prepends an action token and
marks the played square. The default is 192 dims, 6 heads, 4/3/1 blocks
(3.8M parameters). `--dims 256 --heads 8 --repr-layers 6 --dyn-layers 4
--pred-layers 2` (10M) also fits in 16 GB at batch 256.

Throughput on the RTX 5080 laptop GPU (`othello_muzero_torch_bench`):

| model | parameters | inference (positions/s) | training (samples/s, K = 5) | training memory, batch 256 |
|-------|-----------:|------------------------:|----------------------------:|---------------------------:|
| d256, 6/4/2 | 10.0M | 50k | 1.2k | 8.5 GB |
| d192, 6/4/2 | 5.5M | 78k | 1.5k | 6.3 GB |
| d192, 4/3/1 | 3.8M | 117k | 2.2k | 4.5 GB |
| d128, 6/4/1 | 2.3M | 150k | 2.4k | 3.8 GB |

Performance notes:

- Latents stay on the GPU. `ModelInference` keeps them in a ring buffer
  (`--latent-pool`, sized automatically), and the search holds 64-bit
  handles in their place, which it treats as opaque latents. This removed
  32 KB of host copies per evaluation and 8 GB of host RAM.
- Inference and the learner run on separate CUDA streams. Inputs and outputs
  pass through pinned buffers with one synchronization per batch.
- The fused `rms_norm` matters: the composite float32 formula kept float32
  copies of every activation for the backward pass. That pushed training
  past 16 GB, where WSL2 silently pages GPU memory into host RAM and slows
  training 40-fold.
- Learner and self-play share the GPU at the `--sample-reuse` ratio. With
  the defaults the learner is the bottleneck, and self-play waits for it.

### Hyperparameter search

Starting points came from the literature: MiniZero's Othello runs (Wu et al.
2023: batch 1024, 200 simulations for PUCT, Gumbel working with 2-16) and
Pgx's Gumbel AlphaZero (lr 1e-3, 32 simulations). Trials then trained from
scratch at an equal wall clock budget, with in-run evaluation off so every
trial got the same GPU time. Each final model played 100 games against each
fixed opponent in the arena (`scripts/othello_muzero_torch_sweep.sh`,
trial lists in `scripts/sweep_*.txt`). Scores run from 0 to 1.

Stage 1, value targets and learning rate (d192 4/3/1, 15 minutes):

| trial | alphabeta_1 | rollout_1000 | alphabeta_4 |
|-------|------------:|-------------:|------------:|
| value targets from the game result, lr 1e-3 | **0.280** | 0.040 | 0.035 |
| bootstrapped (EfficientZero, horizon 30), lr 1e-3 | 0.040 | 0.010 | 0.010 |
| bootstrapped, lr 3e-4 | 0.100 | 0.000 | 0.000 |

Stage 2, data pipeline and model size (game result targets, 20 minutes):

| trial | steps | games | alphabeta_1 | alphabeta_2 | rollout_1000 | mean |
|-------|------:|------:|------------:|------------:|-------------:|-----:|
| replay window 200k positions | 8616 | 9156 | **0.570** | **0.140** | **0.150** | **0.287** |
| + 32/8 simulations instead of 64/16 | 9333 | 9930 | 0.375 | 0.120 | 0.050 | 0.182 |
| + sample reuse 2 instead of 4 | 7473 | 15895 | 0.285 | 0.100 | 0.150 | 0.178 |
| replay window 1M positions | 8624 | 9158 | 0.255 | 0.040 | 0.000 | 0.098 |
| d256 6/4/2 (10M parameters), window 200k | 4343 | 4574 | 0.080 | 0.010 | 0.000 | 0.030 |

The defaults follow the best trial: d192 4/3/1, game result value targets,
a 200k position replay window, sample reuse 4, 64/16 simulations with full
search probability 0.4, batch 256, Muon 0.02 + AdamW 1e-3. Lessons:

- Bootstrapped value targets hurt early training. They learn from search
  values of a network that cannot evaluate yet.
- A short replay window helps most. A long one keeps training on the
  near-random games of the first minutes.
- More search per move beat more games. Halving the simulations, or the
  training per position, both lost strength.
- The 10M parameter model takes half as many steps in the same time and
  was far behind after 20 minutes. It may catch up in a run of many hours,
  which the search did not test.

Early in training, the value loss stays near the entropy of the outcomes
(about 0.8 per unroll step). That is expected and not a bug: games between
weak players are coin flips until late. An offline check of such a network
calls 84% of the last four positions of held-out games correctly, against
74% for the disc count.

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
| `evaluation_core_test` | the alpha-beta othello baseline: legal moves, pruning matches plain minimax, strength by depth |
| `libtorch_test` | LibTorch `TransformerMuZero` shapes and save/load; `ModelInference` latent pool matching transferred latents, eviction detection; learner fitting and checkpoint resume with Muon and AdamW; a full CPU training run with resume and the time limit |

`test/common/` holds the shared fixtures: tic-tac-toe as a second
`MuZeroGame`, "true models" that use the real rules as the dynamics, small
transformer configurations and random trajectories.
