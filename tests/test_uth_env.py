# tests/test_uth_env.py
"""
Regression tests for the Ultimate Texas Hold'em environment.

Mirrors the table-driven style of tests/test_pokers_regressions.py but runs
on the pure-Python UTH environment (no Rust pokers dependency).

Deal order for UTHState.from_deck: player 2 cards, dealer 2 cards, then the
board is drawn from the top of the remaining deck as needed.
"""

import pytest

from src.envs.uth_env import (
    UTHActionEnum,
    UTHEnv,
    UTHStage,
    UTHState,
    UTHStatus,
    cards_from_strings,
    evaluate_hand,
    make_deck,
)
from src.envs.uth_paytables import (
    FLUSH,
    FULL_HOUSE,
    HIGH_CARD,
    PAIR,
    ROYAL_FLUSH,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    TWO_PAIR,
    FOUR_OF_A_KIND,
    BLIND_PAYTABLE,
    TRIPS_PAYTABLES,
    get_trips_paytable,
)


def deck_for(player, dealer, board):
    """Build a deterministic deck: player 2, dealer 2, board 5, then filler."""
    fixed = cards_from_strings(list(player) + list(dealer) + list(board))
    filler = [c for c in make_deck() if c not in fixed]
    return fixed + filler


def play_through(state, actions):
    for action in actions:
        state = state.apply_action(action)
        assert state.status == UTHStatus.Ok, (
            f"Action {action!r} rejected; legal was {state.legal_actions}"
        )
    return state


# ---------------------------------------------------------------------------
# Hand evaluator
# ---------------------------------------------------------------------------

EVAL_CASES = [
    (["As", "Ks", "Qs", "Js", "Ts", "2c", "3d"], ROYAL_FLUSH),
    (["9s", "Ks", "Qs", "Js", "Ts", "2c", "3d"], STRAIGHT_FLUSH),
    (["Ah", "Ad", "As", "Ac", "Ts", "2c", "3d"], FOUR_OF_A_KIND),
    (["Ah", "Ad", "As", "Tc", "Ts", "2c", "3d"], FULL_HOUSE),
    (["Ah", "Th", "7h", "5h", "2h", "Kc", "3d"], FLUSH),
    (["Ah", "2c", "3d", "4s", "5h", "9c", "Kd"], STRAIGHT),  # wheel
    (["9h", "Tc", "Jd", "Qs", "Kh", "2c", "3d"], STRAIGHT),
    (["Ah", "Ad", "As", "Tc", "9s", "2c", "3d"], THREE_OF_A_KIND),
    (["Ah", "Ad", "Ts", "Tc", "9s", "2c", "3d"], TWO_PAIR),
    (["Ah", "Ad", "Ts", "8c", "9s", "2c", "3d"], PAIR),
    (["Ah", "Kd", "Ts", "8c", "9s", "2c", "3d"], HIGH_CARD),
]


@pytest.mark.parametrize("cards,expected", EVAL_CASES)
def test_evaluate_hand_categories(cards, expected):
    assert evaluate_hand(cards_from_strings(cards))[0] == expected


def test_evaluate_hand_tiebreakers():
    higher = evaluate_hand(cards_from_strings(["Ah", "Ad", "Ks", "8c", "9s", "2c", "3d"]))
    lower = evaluate_hand(cards_from_strings(["Kh", "Kd", "As", "8c", "9s", "2c", "3d"]))
    assert higher > lower  # pair of aces beats pair of kings


# ---------------------------------------------------------------------------
# Determinism / construction
# ---------------------------------------------------------------------------

def test_from_seed_is_deterministic():
    a = UTHState.from_seed(ante=10.0, trips_bet=1.0, stake=1000.0, seed=42)
    b = UTHState.from_seed(ante=10.0, trips_bet=1.0, stake=1000.0, seed=42)
    assert a.player_hand == b.player_hand
    assert a.dealer_hand == b.dealer_hand
    assert a.deck == b.deck


def test_blind_must_equal_ante():
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=10.0, blind=5.0, seed=0)


def test_invalid_paytable_rejected():
    with pytest.raises(ValueError):
        UTHState.from_seed(ante=10.0, seed=0, paytable="UTH-99")
    with pytest.raises(ValueError):
        get_trips_paytable("nope")


# ---------------------------------------------------------------------------
# Stage flow and bet-size constraints
# ---------------------------------------------------------------------------

def test_stage_legal_actions():
    state = UTHState.from_seed(ante=10.0, seed=0)
    assert state.stage == UTHStage.Preflop
    assert set(state.legal_actions) == {
        UTHActionEnum.Check, UTHActionEnum.Bet3x, UTHActionEnum.Bet4x,
    }

    state = state.apply_action(UTHActionEnum.Check)
    assert state.stage == UTHStage.Flop
    assert len(state.public_cards) == 3
    assert set(state.legal_actions) == {UTHActionEnum.Check, UTHActionEnum.Bet2x}

    state = state.apply_action(UTHActionEnum.Check)
    assert state.stage == UTHStage.River
    assert len(state.public_cards) == 5
    assert set(state.legal_actions) == {UTHActionEnum.Fold, UTHActionEnum.Bet1x}


@pytest.mark.parametrize("action,multiple", [
    (UTHActionEnum.Bet3x, 3.0),
    (UTHActionEnum.Bet4x, 4.0),
])
def test_preflop_play_bet_multiples(action, multiple):
    state = UTHState.from_seed(ante=10.0, seed=0).apply_action(action)
    assert state.final_state
    assert state.play_bet == multiple * 10.0


def test_flop_play_bet_is_2x():
    state = UTHState.from_seed(ante=10.0, seed=0)
    state = play_through(state, [UTHActionEnum.Check, UTHActionEnum.Bet2x])
    assert state.final_state
    assert state.play_bet == 2.0 * 10.0


def test_river_play_bet_is_1x():
    state = UTHState.from_seed(ante=10.0, seed=0)
    state = play_through(
        state, [UTHActionEnum.Check, UTHActionEnum.Check, UTHActionEnum.Bet1x]
    )
    assert state.final_state
    assert state.play_bet == 1.0 * 10.0


ILLEGAL_CASES = [
    # (actions to reach stage, illegal action attempted)
    ([], UTHActionEnum.Bet1x),                       # 1x not allowed preflop
    ([], UTHActionEnum.Bet2x),                       # 2x not allowed preflop
    ([], UTHActionEnum.Fold),                        # cannot fold preflop
    ([UTHActionEnum.Check], UTHActionEnum.Bet3x),    # 3x not allowed on flop
    ([UTHActionEnum.Check], UTHActionEnum.Bet4x),    # 4x not allowed on flop
    ([UTHActionEnum.Check], UTHActionEnum.Fold),     # cannot fold on flop
    ([UTHActionEnum.Check, UTHActionEnum.Check], UTHActionEnum.Check),  # river: no check
    ([UTHActionEnum.Check, UTHActionEnum.Check], UTHActionEnum.Bet2x),  # river: no 2x
]


@pytest.mark.parametrize("setup,illegal", ILLEGAL_CASES)
def test_illegal_actions_rejected(setup, illegal):
    state = play_through(UTHState.from_seed(ante=10.0, seed=0), setup)
    result = state.apply_action(illegal)
    assert result.status == UTHStatus.IllegalAction


def test_action_after_final_state_rejected():
    state = UTHState.from_seed(ante=10.0, seed=0).apply_action(UTHActionEnum.Bet4x)
    assert state.final_state
    assert state.apply_action(UTHActionEnum.Check).status == UTHStatus.IllegalAction


# ---------------------------------------------------------------------------
# Settlement regressions (deterministic decks)
# ---------------------------------------------------------------------------

ANTE = 10.0


def test_dealer_not_qualified_ante_pushes():
    """Player wins with a pair; dealer has high card (no qualify).

    Ante pushes, Play pays 1:1, Blind pushes (win below straight)."""
    deck = deck_for(
        player=["Ah", "Kh"],
        dealer=["2c", "7d"],
        board=["As", "9h", "5d", "Jc", "3s"],
    )
    state = UTHState.from_deck(deck, ante=ANTE)
    state = state.apply_action(UTHActionEnum.Bet4x)
    assert state.final_state
    assert state.info["dealer_qualifies"] is False
    assert state.info["ante_net"] == 0.0           # push
    assert state.info["play_net"] == 4 * ANTE      # 1:1 on 4x play
    assert state.info["blind_net"] == 0.0          # win below straight -> push
    assert state.reward == 4 * ANTE


def test_player_wins_with_straight_blind_pays():
    """Player makes a straight vs dealer's qualified pair: Blind pays 1:1."""
    deck = deck_for(
        player=["6h", "7h"],
        dealer=["Kc", "Kd"],
        board=["8s", "9d", "Tc", "2h", "3s"],
    )
    state = UTHState.from_deck(deck, ante=ANTE)
    state = state.apply_action(UTHActionEnum.Bet3x)
    assert state.final_state
    assert state.info["dealer_qualifies"] is True
    assert state.info["player_eval"][0] == STRAIGHT
    assert state.info["ante_net"] == ANTE
    assert state.info["play_net"] == 3 * ANTE
    assert state.info["blind_net"] == BLIND_PAYTABLE[STRAIGHT] * ANTE
    assert state.reward == ANTE + 3 * ANTE + 1.0 * ANTE


def test_player_loses_all_base_bets():
    """Dealer wins with a higher pair: Ante, Play and Blind all lose."""
    deck = deck_for(
        player=["9c", "8d"],
        dealer=["Ac", "Ad"],
        board=["2s", "5h", "9d", "Jc", "Kh"],
    )
    state = UTHState.from_deck(deck, ante=ANTE)
    state = state.apply_action(UTHActionEnum.Bet4x)
    assert state.final_state
    assert state.info["ante_net"] == -ANTE
    assert state.info["play_net"] == -4 * ANTE
    assert state.info["blind_net"] == -ANTE
    assert state.reward == -(ANTE + 4 * ANTE + ANTE)


def test_tie_pushes_everything():
    """Board plays for both (board straight): all base bets push."""
    deck = deck_for(
        player=["2c", "3d"],
        dealer=["2h", "3s"],
        board=["Tc", "Jd", "Qs", "Kh", "Ad"],
    )
    state = UTHState.from_deck(deck, ante=ANTE)
    state = state.apply_action(UTHActionEnum.Bet4x)
    assert state.final_state
    assert state.reward == 0.0


def test_fold_loses_ante_and_blind_but_trips_pays():
    """Trips side bet pays on trips even when the player folds the river."""
    trips_bet = 5.0
    deck = deck_for(
        player=["Ac", "Ad"],
        dealer=["Kc", "Qd"],
        board=["Ah", "5d", "9s", "Jc", "3h"],
    )
    state = UTHState.from_deck(deck, ante=ANTE, trips_bet=trips_bet)
    state = play_through(
        state, [UTHActionEnum.Check, UTHActionEnum.Check, UTHActionEnum.Fold]
    )
    assert state.final_state
    assert state.folded
    assert state.info["player_eval"][0] == THREE_OF_A_KIND
    assert state.info["ante_net"] == -ANTE
    assert state.info["blind_net"] == -ANTE
    assert state.info["play_net"] == 0.0
    trips_payout = TRIPS_PAYTABLES["UTH-01"][THREE_OF_A_KIND] * trips_bet
    assert state.info["trips_net"] == trips_payout
    assert state.reward == -2 * ANTE + trips_payout


def test_trips_loses_when_below_three_of_a_kind():
    trips_bet = 5.0
    deck = deck_for(
        player=["Ah", "Kh"],
        dealer=["2c", "7d"],
        board=["As", "9h", "5d", "Jc", "3s"],  # player has a pair only
    )
    state = UTHState.from_deck(deck, ante=ANTE, trips_bet=trips_bet)
    state = state.apply_action(UTHActionEnum.Bet4x)
    assert state.info["trips_net"] == -trips_bet


def test_trips_paytable_option_is_respected():
    """UTH-02 pays 5:1 on a straight (vs 4:1 in UTH-01)."""
    trips_bet = 2.0
    deck = deck_for(
        player=["6h", "7h"],
        dealer=["Kc", "Kd"],
        board=["8s", "9d", "Tc", "2h", "3s"],
    )
    s1 = UTHState.from_deck(deck, ante=ANTE, trips_bet=trips_bet, paytable="UTH-01")
    s1 = s1.apply_action(UTHActionEnum.Bet3x)
    s2 = UTHState.from_deck(deck, ante=ANTE, trips_bet=trips_bet, paytable="UTH-02")
    s2 = s2.apply_action(UTHActionEnum.Bet3x)
    assert s1.info["trips_net"] == 4.0 * trips_bet
    assert s2.info["trips_net"] == 5.0 * trips_bet


def test_reward_components_sum_and_stake_update():
    state = UTHState.from_seed(ante=ANTE, trips_bet=1.0, stake=1000.0, seed=99)
    final = state.apply_action(UTHActionEnum.Bet4x)
    info = final.info
    assert final.reward == pytest.approx(
        info["ante_net"] + info["blind_net"] + info["play_net"] + info["trips_net"]
    )
    assert final.stake == pytest.approx(1000.0 + final.reward)


# ---------------------------------------------------------------------------
# Gym-style wrapper
# ---------------------------------------------------------------------------

def test_env_step_api():
    env = UTHEnv(ante=10.0, trips_bet=1.0)
    state = env.reset(seed=7)
    assert not state.final_state

    state, reward, done, info = env.step(UTHActionEnum.Check)
    assert reward == 0.0 and not done

    state, reward, done, info = env.step(UTHActionEnum.Check)
    assert reward == 0.0 and not done

    state, reward, done, info = env.step(UTHActionEnum.Bet1x)
    assert done
    assert reward == state.reward
    assert "dealer_qualifies" in info


def test_env_rejects_illegal_action():
    env = UTHEnv(ante=10.0)
    env.reset(seed=7)
    with pytest.raises(ValueError):
        env.step(UTHActionEnum.Bet1x)  # 1x is river-only
