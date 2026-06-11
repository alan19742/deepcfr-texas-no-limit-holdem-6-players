"""Regression tests for the Ultimate Texas Hold'em Deep CFR training pipeline.

Run with:
    python3 -m pytest tests/test_uth_training.py -q

Follows the same patterns as tests/test_training_regressions.py:
TensorBoard is stubbed out with a DummyWriter, evaluation is kept tiny,
and random seeds are fixed for reproducibility.
"""

import os
import random
import sys

import numpy as np
import pytest
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from src.core.uth_deep_cfr import (
    UTH_INPUT_SIZE,
    UTHDeepCFRAgent,
    encode_uth_state,
)
from src.envs.uth_env import NUM_UTH_ACTIONS, UTHActionEnum, UTHState
from src.training import train_uth


class DummyWriter:
    """Stub for torch.utils.tensorboard.SummaryWriter."""

    def __init__(self, *args, **kwargs):
        self.scalars = []

    def add_scalar(self, tag, value, step):
        self.scalars.append((tag, value, step))

    def close(self):
        pass


def _seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

class TestEncoding:
    def test_encoding_shape_and_range(self):
        state = UTHState.from_seed(ante=10.0, trips_bet=1.0, seed=11)
        enc = encode_uth_state(state)
        assert enc.shape == (UTH_INPUT_SIZE,)
        assert np.all(np.isfinite(enc))

    def test_encoding_changes_with_stage(self):
        state = UTHState.from_seed(ante=10.0, seed=12)
        enc_pre = encode_uth_state(state)
        flop_state = state.apply_action(UTHActionEnum.Check)
        enc_flop = encode_uth_state(flop_state)
        assert not np.array_equal(enc_pre, enc_flop)

    def test_encoding_deterministic(self):
        s1 = UTHState.from_seed(ante=10.0, seed=13)
        s2 = UTHState.from_seed(ante=10.0, seed=13)
        assert np.array_equal(encode_uth_state(s1), encode_uth_state(s2))


# ---------------------------------------------------------------------------
# Agent internals
# ---------------------------------------------------------------------------

class TestUTHDeepCFRAgent:
    def test_traverse_populates_memories(self):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        for game in range(5):
            state = UTHState.from_seed(ante=10.0, trips_bet=1.0, seed=game)
            agent.cfr_traverse(state, iteration=1)
        assert len(agent.advantage_memory) > 0
        assert len(agent.strategy_memory) > 0

        encoded, legal_mask, regrets, iteration = agent.advantage_memory[0]
        assert encoded.shape == (UTH_INPUT_SIZE,)
        assert legal_mask.shape == (NUM_UTH_ACTIONS,)
        assert regrets.shape == (NUM_UTH_ACTIONS,)
        assert iteration == 1
        # Regrets must be zero for illegal actions
        assert np.all(regrets[legal_mask == 0] == 0)

    def test_traverse_ev_is_finite_and_bounded(self):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        for game in range(10):
            state = UTHState.from_seed(ante=10.0, trips_bet=1.0, seed=game)
            ev = agent.cfr_traverse(state, iteration=1)
            assert np.isfinite(ev)
            # Max loss: ante + blind + 4x play + trips
            assert -61.0 <= ev <= 600.0

    def test_training_returns_finite_loss(self):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        for game in range(10):
            state = UTHState.from_seed(ante=10.0, trips_bet=1.0, seed=game)
            agent.cfr_traverse(state, iteration=1)
        adv_loss = agent.train_advantage_network(batch_size=32, epochs=2)
        strat_loss = agent.train_strategy_network(batch_size=32, epochs=2)
        assert np.isfinite(adv_loss)
        assert np.isfinite(strat_loss)

    def test_choose_action_is_always_legal(self):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        for seed in range(20):
            state = UTHState.from_seed(ante=10.0, seed=seed)
            steps = 0
            while not state.final_state and steps < 10:
                action = agent.choose_action(state)
                assert action in state.legal_actions
                state = state.apply_action(action)
                steps += 1
            assert state.final_state

    def test_save_and_load_checkpoint(self, tmp_path):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        state = UTHState.from_seed(ante=10.0, seed=0)
        agent.cfr_traverse(state, iteration=1)
        agent.train_advantage_network(batch_size=16, epochs=1)
        agent.train_strategy_network(batch_size=16, epochs=1)
        agent.iteration_count = 7

        prefix = str(tmp_path / "uth_checkpoint")
        agent.save_model(prefix)
        assert os.path.exists(prefix + ".pt")

        fresh = UTHDeepCFRAgent(device="cpu")
        fresh.load_model(prefix + ".pt")
        assert fresh.iteration_count == 7

        # Same weights -> identical strategy output
        s = UTHState.from_seed(ante=10.0, seed=5)
        enc = torch.tensor(encode_uth_state(s), dtype=torch.float32).unsqueeze(0)
        with torch.no_grad():
            out_a, _ = agent.strategy_net(enc)
            out_b, _ = fresh.strategy_net(enc)
        assert torch.allclose(out_a, out_b)


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

class TestEvaluation:
    def test_evaluate_agent_returns_finite_profit(self):
        _seed_everything(0)
        agent = UTHDeepCFRAgent(device="cpu")
        avg_profit = train_uth.evaluate_agent(
            agent, num_games=20, ante=10.0, trips_bet=1.0
        )
        assert np.isfinite(avg_profit)

    def test_random_baseline_returns_finite_profit(self):
        avg_profit = train_uth.evaluate_baseline_random(
            num_games=50, ante=10.0, trips_bet=1.0
        )
        assert np.isfinite(avg_profit)
        # House edge exists but a random policy can't lose more than the
        # maximum total wager per hand.
        assert -61.0 <= avg_profit <= 100.0


# ---------------------------------------------------------------------------
# Training pipeline smoke tests
# ---------------------------------------------------------------------------

class TestTrainUTH:
    def test_training_loop_smoke(self, tmp_path):
        """One tiny iteration end-to-end: memories fill, checkpoints saved."""
        _seed_everything(0)
        writer = DummyWriter()
        save_dir = str(tmp_path / "models")
        agent, profits = train_uth.train_uth_deep_cfr(
            iterations=1,
            traversals_per_iteration=5,
            ante=10.0,
            trips_bet=1.0,
            save_dir=save_dir,
            eval_every=1,
            eval_games=10,
            seed=0,
            verbose=False,
            writer=writer,
        )
        assert len(agent.advantage_memory) > 0
        assert len(agent.strategy_memory) > 0
        assert len(profits) == 1 and np.isfinite(profits[0])
        files = os.listdir(save_dir)
        assert any(f.endswith(".pt") for f in files)
        # TensorBoard scalars were recorded through the injected writer
        tags = {tag for tag, _, _ in writer.scalars}
        assert "Loss/Advantage" in tags
        assert "Performance/ProfitPerHand" in tags

    def test_checkpoint_resume(self, tmp_path):
        _seed_everything(0)
        save_dir = str(tmp_path / "models")
        train_uth.train_uth_deep_cfr(
            iterations=1,
            traversals_per_iteration=3,
            save_dir=save_dir,
            eval_every=10,
            eval_games=5,
            seed=0,
            verbose=False,
            writer=DummyWriter(),
        )
        final_path = os.path.join(save_dir, "uth_final.pt")
        assert os.path.exists(final_path)

        fresh = UTHDeepCFRAgent(device="cpu")
        fresh.load_model(final_path)
        state = UTHState.from_seed(ante=10.0, seed=99)
        assert fresh.choose_action(state) in state.legal_actions

    def test_training_is_reproducible(self, tmp_path):
        """Same seed -> same first advantage memory contents."""
        def run(tag):
            _seed_everything(123)
            agent, _ = train_uth.train_uth_deep_cfr(
                iterations=1,
                traversals_per_iteration=3,
                save_dir=str(tmp_path / tag),
                eval_every=10,
                eval_games=2,
                seed=123,
                verbose=False,
                writer=DummyWriter(),
            )
            return agent

        a1 = run("a")
        a2 = run("b")
        e1, m1, r1, _ = a1.advantage_memory[0]
        e2, m2, r2, _ = a2.advantage_memory[0]
        assert np.array_equal(e1, e2)
        assert np.array_equal(m1, m2)
        assert np.allclose(r1, r2)


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-q"]))
