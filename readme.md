# DeepCFR Poker AI

Deep CFR for 6-player no-limit Texas Hold'em, built on top of the `pokers` environment and focused on practical training workflows from source.

![Poker AI](https://raw.githubusercontent.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players/refs/heads/main/images/testing_different_iteration_values/Screenshot%202025-03-04%20at%2014.39.24.png)

## Project Update (March 2026)

This repo has moved past the March 2025 state described in the original article and early README.

Current status:

- The project is source-first and ready to train from the repository.
- The all-in edge cases in the underlying poker engine are handled by pinning to the patched `pokers` fork in [requirements.txt](./requirements.txt).
- The Phase 2 self-play and Phase 3 mixed-training regressions from issue `#22` have been fixed on `main`.
- Regression coverage now exists for both poker-engine integration and training-path smoke tests.

What this means in practice:

- Basic random-opponent training works.
- Continuing training from a checkpoint works.
- Self-play against a checkpoint works.
- Mixed checkpoint training works again.
- Opponent-modeling training scripts are still available, but should be treated as experimental compared with the main training path.

The Medium article is still useful for background, but the code has evolved. Prefer this README and the current scripts over the article when they differ.

## Installation

Recommended workflow: run directly from source.

```bash
git clone https://github.com/dberweger2017/deepcfr-texas-no-limit-holdem-6-players.git
cd deepcfr-texas-no-limit-holdem-6-players

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Notes:

- The GUI requires `PyQt5`, which is already included in [requirements.txt](./requirements.txt).
- The documented commands below use `python -m ...` or `python scripts/...` from the repo root. That is the maintained workflow.

## What Works Today

- Deep CFR training against random opponents
- Checkpoint continuation
- Self-play training against a fixed checkpoint
- Mixed training against a rotating checkpoint pool
- Opponent-modeling training variants
- CLI play against saved checkpoints or random agents
- PyQt GUI play
- Tournament visualization across checkpoints
- Regression tests for known `pokers` and training-path failures

## Architecture

The current implementation uses:

- A 6-player no-limit Texas Hold'em environment from [`pokers`](https://github.com/Reinforcement-Poker/pokers)
- A fixed-length state encoding including hole cards, board cards, stage, pot, positions, player states, min bet, legal actions, and previous action
- A shared feed-forward network body with two heads:
  - action head: `Fold`, `Check/Call`, `Raise`
  - sizing head: continuous raise sizing in roughly `0.1x` to `3.0x` pot
- Prioritized replay for advantage training
- Separate strategy-memory storage for policy updates
- Optional opponent-modeling variants built around GRU-based action-history encoding

This repo no longer uses the older 4-action "half-pot / pot raise" architecture described in earlier versions of the README. The current action model is 3 action types plus continuous raise sizing.

## Training

All commands below are run from the repository root.

### Phase 1: Train Against Random Opponents

```bash
python -m src.training.train --iterations 1000 --traversals 200 --log-dir logs/phase1 --save-dir models/phase1
```

### Continue Training From a Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/phase1/checkpoint_iter_1000.pt \
  --iterations 1000 \
  --traversals 200 \
  --log-dir logs/continued \
  --save-dir models/continued
```

### Phase 2: Self-Play Against a Fixed Checkpoint

```bash
python -m src.training.train \
  --checkpoint models/phase1/checkpoint_iter_1000.pt \
  --self-play \
  --iterations 2000 \
  --traversals 400 \
  --log-dir logs/selfplay \
  --save-dir models/selfplay
```

### Phase 3: Mixed Training Against a Checkpoint Pool

```bash
python -m src.training.train \
  --mixed \
  --checkpoint-dir models \
  --model-prefix t_ \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --iterations 10000 \
  --traversals 400 \
  --log-dir logs/mixed \
  --save-dir models/mixed
```

### Opponent-Modeling Training

Basic opponent-modeling training:

```bash
python -m src.training.train_with_opponent_modeling \
  --iterations 1000 \
  --traversals 200 \
  --save-dir models_om \
  --log-dir logs/deepcfr_om
```

Mixed opponent-modeling training:

```bash
python -m src.training.train_mixed_with_opponent_modeling \
  --checkpoint-dir models_om \
  --model-prefix "*" \
  --iterations 10000 \
  --traversals 200 \
  --refresh-interval 1000 \
  --num-opponents 5 \
  --save-dir models_mixed_om \
  --log-dir logs/deepcfr_mixed_om
```

### Monitor Training

```bash
tensorboard --logdir=logs
```

Then open `http://localhost:6006`.

## Playing Against the Models

### CLI

```bash
python scripts/play.py --models-dir models/phase1
```

Useful options:

- `--model-pattern "*.pt"` to filter checkpoint files
- `--num-models 5` to control how many checkpoint opponents are sampled
- `--position 0` to choose your seat
- `--no-shuffle` to keep the same sampled models across games
- `--strict` to raise on invalid game states instead of logging and continuing

### GUI

```bash
python scripts/poker_gui.py --models_folder models/phase1
```

### Tournament Visualization

```bash
python scripts/visualize_tournament.py \
  --checkpoints models/phase1/checkpoint_iter_1000.pt models/selfplay/checkpoint_iter_2000.pt \
  --num-games 100
```

## Ultimate Texas Hold'em (UTH)

The repo also includes a pure-Python Ultimate Texas Hold'em environment and a Deep CFR training pipeline for it. UTH is a heads-up game against the house: the player posts equal Ante and Blind bets (plus an optional Trips side bet), then has one chance to raise — 3x/4x ante pre-flop, 2x on the flop, or 1x on the river (otherwise fold). The dealer needs a pair or better to qualify; the Blind bet pays a bonus table from a straight up, and Trips pays on three of a kind or better regardless of who wins.

Key files:

- `src/envs/uth_env.py` — game engine (`UTHState`, `UTHEnv`), 7-card hand evaluator, settlement logic
- `src/envs/uth_paytables.py` — configurable Blind/Trips paytables (UTH-01 .. UTH-04, default UTH-01)
- `src/core/uth_deep_cfr.py` — `UTHDeepCFRAgent` with state encoding over the 6-way action space (Fold, Check, Bet1x..Bet4x)
- `src/training/train_uth.py` — training loop with TensorBoard logging, periodic eval, and checkpoints

The environment does not require the Rust `pokers` package — it runs anywhere Python + PyTorch run.

### Interactive game interface

The easiest way to start the web interface (auto-installs web deps if needed):

```bash
bash start_game.sh
# then open http://localhost:8000 in your browser
```

To change the port:

```bash
UTH_PORT=9000 bash start_game.sh
```

Manual start (if you manage the venv yourself):

```bash
# step 1 — install web dependencies (one time only)
pip install -r requirements-web.txt

# step 2 — start the server
python3 -m uvicorn web.uth_server:app --host 0.0.0.0 --port 8000 --reload

# step 3 — open http://localhost:8000
```

The server watches for file changes (`--reload`) so edits to the engine or frontend are picked up without a restart.

### Train

```bash
python3 -m src.training.train_uth --iterations 200 --traversals-per-iteration 200 \
  --ante 10 --trips-bet 1 --paytable UTH-01 --save-dir models_uth --verbose
# or, after `pip install -e .`:
deepcfr-train-uth --iterations 200
```

### Test

```bash
python3 -m pytest tests/test_uth_env.py tests/test_uth_training.py -q
```

## Testing and Regression Coverage

The repo now includes targeted regression tests for the issues that have caused the most damage recently.

Run them with:

```bash
python3 -m pytest tests/test_pokers_regressions.py tests/test_training_regressions.py -q
```

What these cover:

- `tests/test_pokers_regressions.py`
  - all-in and legal-action regressions inherited from the `pokers` library
- `tests/test_training_regressions.py`
  - self-play and mixed-training smoke tests
  - replay-memory shape consistency
  - explicit `.pt` save-path handling
- `tests/test_uth_env.py`
  - UTH rule regressions: bet-sizing caps per street, dealer qualification, Blind/Trips paytable settlement, determinism, money conservation
- `tests/test_uth_training.py`
  - UTH Deep CFR traversal/training smoke tests, checkpoint save/load, reproducibility

## Notes on Results

Some older README claims and article screenshots implied a more stable training outcome than the current repo can honestly guarantee.

What is safe to say today:

- the main training paths run
- the known training-path bugs from issue `#22` are fixed
- the all-in legal-action bugs that were breaking games are fixed in the pinned `pokers` fork

What is still an open research / tuning question:

- exact profitability numbers versus the article
- how robust the learned strategy is across seeds and training schedules
- whether opponent-modeling variants outperform the simpler baseline consistently

If you care about reproducibility, run multiple seeds and compare checkpoints rather than relying on a single training curve.

## Future Work

The forward-looking backlog lives in [FUTURE_IMPROVEMENTS.md](./FUTURE_IMPROVEMENTS.md). It has been trimmed to items that still make sense after the recent architecture and training fixes.

## References

1. Brown, N., and Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164).
2. Zinkevich, M., Johanson, M., Bowling, M., and Piccione, C. (2008). [Regret Minimization in Games with Incomplete Information](https://papers.nips.cc/paper/3306-regret-minimization-in-games-with-incomplete-information.pdf).
3. Heinrich, J., and Silver, D. (2016). [Deep Reinforcement Learning from Self-Play in Imperfect-Information Games](https://arxiv.org/abs/1603.01121).

## License

This project is licensed under the MIT License. See [LICENSE.txt](./LICENSE.txt).

## Acknowledgments

- The maintainers of [`pokers`](https://github.com/Reinforcement-Poker/pokers)
- The community members who reported and reproduced training and game-state bugs
- The PyTorch ecosystem for making iteration on this kind of project practical
