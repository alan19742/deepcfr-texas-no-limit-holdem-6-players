# tests/test_uth_training_smoke.py
"""
Training smoke tests for the UTH Deep CFR agent (mirrors
tests/test_training_regressions.py): a few iterations with few traversals,
asserting finite losses, consistent replay-memory shapes, and checkpoint
save / reload round-trips.
"""

import math
import os

import numpy as np
import pytest
import torch

from src.core.uth_agent import UTHDeepCFRAgent, encode_uth_state
from src.core.uth_env import UTHState, UTHStatus, UTHAction, UTHActionEnum
from src.training.train_uth import evaluate_uth
from src.utils.settings import set_strict_checking


@pytest.fixture(autouse=True)
def strict_mode():
    set_strict_checking(True)
    yield
    set_strict_checking(False)


@pytest.fixture
def agent():
    torch.manual_seed(0)
    np.random.seed(0)
    return UTHDeepCFRAgent(player_id=0, device='cpu')


def run_traversals(agent, iteration, count, seed_base=0, bonus_bet=0.0):
    for t in range(count):
        state = UTHState.from_seed(ante=10.0, bonus_bet=bonus_bet,
                                   stake=1000.0, seed=seed_base + t)
        agent.cfr_traverse(state, iteration)


def test_encoding_shape_consistency(agent):
    state = UTHState.from_seed(ante=10.0, bonus_bet=5.0, stake=1000.0, seed=1)
    enc = encode_uth_state(state)
    assert enc.shape == (agent.input_size,)

    flop = state.apply_action(UTHAction(UTHActionEnum.Check))
    assert encode_uth_state(flop).shape == (agent.input_size,)

    river = flop.apply_action(UTHAction(UTHActionEnum.Check))
    assert encode_uth_state(river).shape == (agent.input_size,)


def test_traversal_fills_memories_with_consistent_shapes(agent):
    run_traversals(agent, iteration=1, count=30)

    assert len(agent.advantage_memory) > 0
    assert len(agent.strategy_memory) > 0

    batch, _, _ = agent.advantage_memory.sample(
        min(16, len(agent.advantage_memory)), beta=0.4
    )
    for state_enc, opp_feats, action_type, bet_size, regret in batch:
        assert state_enc.shape == (agent.input_size,)
        assert opp_feats.shape == (20,)
        assert action_type in (0, 1, 2)
        assert math.isfinite(float(bet_size))
        assert math.isfinite(float(regret))

    for state_enc, opp_feats, strategy, bet_size, iteration in list(agent.strategy_memory)[:16]:
        assert state_enc.shape == (agent.input_size,)
        assert strategy.shape == (agent.num_actions,)
        assert abs(float(np.sum(strategy)) - 1.0) < 1e-5
        assert math.isfinite(float(bet_size))


def test_training_loop_smoke(agent):
    losses = []
    for iteration in range(1, 4):
        agent.iteration_count = iteration
        run_traversals(agent, iteration, count=40,
                       seed_base=iteration * 1000, bonus_bet=5.0)
        loss = agent.train_advantage_network(batch_size=32, epochs=2)
        losses.append(loss)

    strat_loss = agent.train_strategy_network(batch_size=32, epochs=2)

    assert all(math.isfinite(l) for l in losses)
    assert any(l > 0 for l in losses), "advantage network never trained"
    assert math.isfinite(strat_loss)


def test_checkpoint_save_and_reload(agent, tmp_path):
    run_traversals(agent, iteration=1, count=30)
    agent.iteration_count = 1
    agent.train_advantage_network(batch_size=32, epochs=1)

    prefix = os.path.join(str(tmp_path), "uth_model")
    agent.save_model(prefix)
    saved_path = f"{prefix}_iteration_1.pt"
    assert os.path.exists(saved_path)

    fresh = UTHDeepCFRAgent(player_id=0, device='cpu')
    fresh.load_model(saved_path)
    assert fresh.iteration_count == 1

    # Weights actually round-trip.
    for (k1, v1), (k2, v2) in zip(agent.strategy_net.state_dict().items(),
                                  fresh.strategy_net.state_dict().items()):
        assert k1 == k2
        assert torch.equal(v1, v2)

    # Reloaded agent can keep training.
    run_traversals(fresh, iteration=2, count=20, seed_base=5000)
    loss = fresh.train_advantage_network(batch_size=32, epochs=1)
    assert math.isfinite(loss)


def test_evaluate_uth_returns_finite(agent):
    profit = evaluate_uth(agent, num_games=50, ante=10.0, bonus_bet=0.0,
                          stake=1000.0)
    assert math.isfinite(profit)
    # Per-ante units: bounded by worst case (lose ante+blind+4x play+trips).
    assert -7.0 <= profit <= 600.0


def test_choose_action_always_legal(agent):
    import random as _random
    rng = _random.Random(9)
    for seed in range(50):
        state = UTHState.from_seed(ante=10.0, bonus_bet=rng.choice([0.0, 5.0]),
                                   stake=1000.0, seed=seed)
        guard = 0
        while not state.final_state and guard < 10:
            action = agent.choose_action(state)
            state = state.apply_action(action)
            assert state.status == UTHStatus.Ok
            guard += 1
        assert state.final_state
