# tests/test_uth_env.py
"""
Environment / engine smoke tests for the UTH state machine:
  - seed reproducibility
  - random rollouts never leave Ok status and produce finite rewards
  - the pure-Python evaluator agrees with the `pokers` Rust evaluator on
    random 7-card samples (cross-validated through heads-up showdowns)
"""

import math
import random

import pytest
import pokers as pkrs

from src.core.uth_env import (
    UTHAction,
    UTHActionEnum,
    UTHState,
    UTHStatus,
    UTHCard,
    evaluate_seven,
    full_deck,
)


def random_rollout(state, rng):
    """Play random legal actions until the hand finishes."""
    guard = 0
    while not state.final_state and guard < 10:
        action_enum = rng.choice(state.legal_actions)
        if action_enum == UTHActionEnum.Raise:
            mult = rng.choice(state.legal_play_multipliers())
            action = UTHAction(action_enum, mult)
        else:
            action = UTHAction(action_enum)
        state = state.apply_action(action)
        guard += 1
    return state


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------

def test_from_seed_reproducible():
    a = UTHState.from_seed(ante=10.0, bonus_bet=1.0, stake=1000.0, seed=42)
    b = UTHState.from_seed(ante=10.0, bonus_bet=1.0, stake=1000.0, seed=42)
    assert a.players_state[0].hand == b.players_state[0].hand
    assert a.players_state[1].hand == b.players_state[1].hand
    assert a.board == b.board

    # Same seed + same action sequence -> same reward.
    rng1, rng2 = random.Random(7), random.Random(7)
    fa, fb = random_rollout(a, rng1), random_rollout(b, rng2)
    assert fa.reward == fb.reward


def test_different_seeds_differ():
    hands = {
        tuple((c.suit, c.rank) for c in
              UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=1000.0,
                                 seed=s).players_state[0].hand)
        for s in range(20)
    }
    assert len(hands) > 1


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------

def test_invalid_constructor_args():
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=0.0, bonus_bet=0.0, stake=1000.0, seed=0)
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=10.0, bonus_bet=-1.0, stake=1000.0, seed=0)
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=30.0, seed=0)
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=1000.0, seed=0,
                           paytable="UTH-99")


# ---------------------------------------------------------------------------
# Random rollouts: status stays Ok, rewards finite
# ---------------------------------------------------------------------------

def test_random_rollouts_clean():
    rng = random.Random(123)
    for seed in range(300):
        state = UTHState.from_seed(ante=10.0, bonus_bet=rng.choice([0.0, 5.0]),
                                   stake=1000.0, seed=seed)
        final = random_rollout(state, rng)
        assert final.final_state, f"seed {seed} did not finish"
        assert final.status == UTHStatus.Ok, f"seed {seed} status {final.status}"
        assert math.isfinite(final.reward), f"seed {seed} reward {final.reward}"
        # Zero-sum between player and dealer rows.
        assert final.players_state[0].reward == -final.players_state[1].reward


def test_terminal_state_rejects_actions():
    state = UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=1000.0, seed=0)
    final = state.apply_action(UTHAction(UTHActionEnum.Raise, 4.0))
    assert final.final_state
    again = final.apply_action(UTHAction(UTHActionEnum.Raise, 4.0))
    assert again.final_state
    assert again.reward == final.reward


def test_stage_progression_and_reveals():
    state = UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=1000.0, seed=5)
    assert len(state.public_cards) == 0
    flop = state.apply_action(UTHAction(UTHActionEnum.Check))
    assert len(flop.public_cards) == 3
    river = flop.apply_action(UTHAction(UTHActionEnum.Check))
    assert len(river.public_cards) == 5
    done = river.apply_action(UTHAction(UTHActionEnum.Raise, 1.0))
    assert done.final_state and done.status == UTHStatus.Ok


# ---------------------------------------------------------------------------
# Evaluator cross-validation against the pokers Rust engine
# ---------------------------------------------------------------------------

def _pkrs_card(c):
    """Map a UTHCard to the matching pokers deck card (idx = suit*13+rank)."""
    return pkrs.Card.collect()[c.suit * 13 + c.rank]


def _is_ace_low_wheel(result):
    """True when the best hand is an ace-low (5-high) straight/straight flush."""
    from src.core.uth_env import HandRank
    return (result[0] in (HandRank.STRAIGHT, HandRank.STRAIGHT_FLUSH)
            and result[1][0] == 3)  # rank index 3 == card '5'


def test_evaluator_matches_pokers_on_random_samples():
    """
    Compare hand strength *ordering* with pokers: deal two 2-card hands on a
    shared 5-card board via State.from_deck heads-up check-down, and compare
    the showdown winner with our evaluate_seven comparison. 1000 samples.

    Known pokers (Rust) bug: the ace-low wheel (A-2-3-4-5) is mis-ranked as
    an ace-HIGH straight, so e.g. a 7-high straight vs a wheel is scored as a
    tie. Trials where either hand makes a wheel are skipped here; our own
    wheel ordering is asserted separately in test_wheel_straight_ordering.
    """
    rng = random.Random(2024)
    deck_template = full_deck()
    mismatches = 0
    skipped_wheels = 0

    for trial in range(1000):
        deck = list(deck_template)
        rng.shuffle(deck)
        # pokers heads-up deal order (from_deck): p0 two, p1 two, then board.
        p0 = deck[0:2]
        p1 = deck[2:4]
        board = deck[4:9]

        ours0 = evaluate_seven(p0 + board)
        ours1 = evaluate_seven(p1 + board)
        if _is_ace_low_wheel(ours0) or _is_ace_low_wheel(ours1):
            skipped_wheels += 1
            continue
        our_sign = (ours0 > ours1) - (ours0 < ours1)

        pk_deck = [_pkrs_card(c) for c in deck]
        state = pkrs.State.from_deck(n_players=2, button=0, sb=1.0, bb=2.0,
                                     stake=100.0, deck=pk_deck)
        guard = 0
        while not state.final_state and guard < 20:
            legal = state.legal_actions
            if pkrs.ActionEnum.Check in legal:
                a = pkrs.Action(pkrs.ActionEnum.Check)
            else:
                a = pkrs.Action(pkrs.ActionEnum.Call)
            state = state.apply_action(a)
            guard += 1

        assert int(state.status) == int(pkrs.StateStatus.Ok)

        # Identify which pokers player got which hand (button/blind order may
        # rotate the deal), then compare rewards.
        def hand_key(ps):
            return tuple(sorted((int(c.suit), int(c.rank)) for c in ps.hand))

        key0 = tuple(sorted((c.suit, c.rank) for c in p0))
        if hand_key(state.players_state[0]) == key0:
            r0, r1 = state.players_state[0].reward, state.players_state[1].reward
        else:
            r0, r1 = state.players_state[1].reward, state.players_state[0].reward

        pk_sign = (r0 > r1) - (r0 < r1)
        if our_sign != pk_sign:
            mismatches += 1

    assert mismatches == 0, f"{mismatches}/1000 evaluator disagreements with pokers"
    # Sanity: the skip path shouldn't swallow a large share of the samples.
    assert skipped_wheels < 100


def test_wheel_straight_ordering():
    """The ace-low wheel must lose to any higher straight (documents the
    pokers Rust bug excluded from the cross-validation above)."""
    from src.core.uth_env import card

    board = [card(t) for t in ("5c", "2h", "Ac", "4c", "3s")]
    seven_high = evaluate_seven([card("6h"), card("7d")] + board)  # 3-7 straight
    wheel = evaluate_seven([card("Td"), card("2c")] + board)       # A-5 wheel

    from src.core.uth_env import HandRank
    assert seven_high[0] == HandRank.STRAIGHT
    assert wheel[0] == HandRank.STRAIGHT
    assert seven_high > wheel
