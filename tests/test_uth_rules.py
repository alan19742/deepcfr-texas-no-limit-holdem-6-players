# tests/test_uth_rules.py
"""
Rule-correctness tests for the Ultimate Texas Hold'em settlement engine.

All cases use constructed hands (no randomness) against the pure function
`settle_uth` plus the hand evaluator, mirroring the table-driven style of
tests/test_pokers_regressions.py.
"""

import pytest

from src.core.uth_env import (
    DEFAULT_PAYTABLE,
    PAYTABLES,
    HandRank,
    UTHStage,
    UTHState,
    UTHStatus,
    UTHAction,
    UTHActionEnum,
    card,
    dealer_qualifies,
    evaluate_seven,
    settle_uth,
)


def cards(*texts):
    return [card(t) for t in texts]


# ---------------------------------------------------------------------------
# Hand evaluator sanity
# ---------------------------------------------------------------------------

EVAL_CASES = [
    pytest.param(cards("As", "Ks", "Qs", "Js", "Ts", "2c", "3d"),
                 HandRank.ROYAL_FLUSH, id="royal-flush"),
    pytest.param(cards("9h", "8h", "7h", "6h", "5h", "Ac", "Ad"),
                 HandRank.STRAIGHT_FLUSH, id="straight-flush"),
    pytest.param(cards("Ah", "5h", "4h", "3h", "2h", "Kc", "Kd"),
                 HandRank.STRAIGHT_FLUSH, id="steel-wheel-ace-low"),
    pytest.param(cards("Qc", "Qd", "Qh", "Qs", "2c", "3d", "4h"),
                 HandRank.FOUR_OF_A_KIND, id="quads"),
    pytest.param(cards("Jc", "Jd", "Jh", "8s", "8c", "2d", "3h"),
                 HandRank.FULL_HOUSE, id="full-house"),
    pytest.param(cards("Kd", "Td", "7d", "4d", "2d", "Ac", "As"),
                 HandRank.FLUSH, id="flush"),
    pytest.param(cards("9c", "8d", "7h", "6s", "5c", "2d", "2h"),
                 HandRank.STRAIGHT, id="straight"),
    pytest.param(cards("Ac", "2d", "3h", "4s", "5c", "9d", "Th"),
                 HandRank.STRAIGHT, id="wheel-straight"),
    pytest.param(cards("7c", "7d", "7h", "Ks", "2c", "3d", "9h"),
                 HandRank.THREE_OF_A_KIND, id="trips"),
    pytest.param(cards("7c", "7d", "Kh", "Ks", "2c", "3d", "9h"),
                 HandRank.TWO_PAIR, id="two-pair"),
    pytest.param(cards("7c", "7d", "Kh", "Qs", "2c", "3d", "9h"),
                 HandRank.PAIR, id="pair"),
    pytest.param(cards("Ac", "Kd", "Qh", "Js", "9c", "3d", "2h"),
                 HandRank.HIGH_CARD, id="high-card"),
]


@pytest.mark.parametrize("seven, expected_rank", EVAL_CASES)
def test_evaluate_seven_rank(seven, expected_rank):
    assert evaluate_seven(seven)[0] == expected_rank


# ---------------------------------------------------------------------------
# Dealer qualification: Ante pushes when dealer doesn't qualify
# ---------------------------------------------------------------------------

def test_dealer_not_qualified_ante_pushes_not_auto_win():
    # Board has no pair; dealer holds unconnected high cards -> no pair.
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    player = cards("As", "Qs")   # Ace-high beats dealer Q-high... use weaker dealer
    dealer = cards("3s", "7c")   # 7-high board only: K-high for dealer
    assert not dealer_qualifies(dealer, board)

    # Player wins with high card (below straight): Ante pushes (dealer not
    # qualified), Play pays 1:1, Blind pushes (win below straight).
    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=0,
                        play=10, folded=False)
    assert reward == 10.0  # Play only; Ante push is NOT an automatic win


def test_dealer_not_qualified_player_loses_still_loses_play_blind():
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    player = cards("3s", "7c")   # K-high with weak kickers
    dealer = cards("As", "Qs")   # K-high w/ A,Q kickers -> dealer wins, no pair
    assert not dealer_qualifies(dealer, board)

    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=0,
                        play=10, folded=False)
    # Ante pushes, Play loses, Blind loses.
    assert reward == -20.0


def test_dealer_qualified_player_wins_below_straight():
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    player = cards("Ks", "Qs")   # pair of kings
    dealer = cards("9s", "3c")   # pair of nines -> qualifies
    assert dealer_qualifies(dealer, board)

    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=0,
                        play=10, folded=False)
    # Ante +10, Play +10, Blind push (win below straight).
    assert reward == 20.0


def test_dealer_qualified_player_loses_all():
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    player = cards("9s", "3c")   # pair of nines
    dealer = cards("Ks", "Qs")   # pair of kings
    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=0,
                        play=10, folded=False)
    assert reward == -30.0


def test_tie_pushes_everything():
    board = cards("As", "Kd", "Qh", "Jc", "Tc")  # board plays: broadway
    player = cards("2c", "3d")
    dealer = cards("2h", "3s")
    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=0,
                        play=10, folded=False)
    assert reward == 0.0


# ---------------------------------------------------------------------------
# Blind paytable (UTH-01; Blind table identical across all four)
# ---------------------------------------------------------------------------

BLIND_CASES = [
    # (player, dealer, board, expected blind multiple)
    pytest.param(cards("As", "Ks"), cards("2c", "2d"),
                 cards("Qs", "Js", "Ts", "3h", "4h"), 500.0, id="royal-500x"),
    pytest.param(cards("9h", "8h"), cards("Ac", "Ad"),
                 cards("7h", "6h", "5h", "Kc", "2s"), 50.0, id="sf-50x"),
    pytest.param(cards("Qc", "Qd"), cards("Ac", "Kd"),
                 cards("Qh", "Qs", "2c", "3d", "9h"), 10.0, id="quads-10x"),
    pytest.param(cards("Jc", "Jd"), cards("Ac", "Ad"),
                 cards("Jh", "8s", "8c", "2d", "3h"), 3.0, id="fullhouse-3x"),
    pytest.param(cards("Kd", "Td"), cards("Ac", "As"),
                 cards("7d", "4d", "2d", "Js", "Jc"), 1.5, id="flush-3to2"),
    pytest.param(cards("9c", "8d"), cards("2s", "2h"),
                 cards("7h", "6s", "5c", "Kd", "2c"), 1.0, id="straight-1x"),
]


@pytest.mark.parametrize("player, dealer, board, blind_mult", BLIND_CASES)
def test_blind_paytable_on_win(player, dealer, board, blind_mult):
    ante = 10.0
    reward = settle_uth(player, dealer, board, ante=ante, blind=ante,
                        trips=0, play=ante, folded=False)
    player_best = evaluate_seven(list(player) + list(board))
    dealer_best = evaluate_seven(list(dealer) + list(board))
    assert player_best > dealer_best, "test case must be a player win"

    expected = ante * blind_mult + ante  # blind + play 1:1
    if dealer_qualifies(dealer, board):
        expected += ante  # ante 1:1
    assert reward == pytest.approx(expected)


# ---------------------------------------------------------------------------
# Trips side bet
# ---------------------------------------------------------------------------

def test_trips_pays_even_on_fold():
    board = cards("7h", "7d", "2c", "9s", "Kd")
    player = cards("7c", "3d")   # trips sevens
    dealer = cards("Ac", "Ad")
    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=5,
                        play=0, folded=True)
    # Fold loses ante+blind (-20); Trips pays 3:1 on three of a kind (+15).
    assert reward == -20.0 + 5 * 3.0


def test_trips_loses_without_three_of_a_kind():
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    player = cards("Ks", "Qs")   # one pair only
    dealer = cards("9s", "3c")
    with_trips = settle_uth(player, dealer, board, ante=10, blind=10,
                            trips=5, play=10, folded=False)
    without = settle_uth(player, dealer, board, ante=10, blind=10,
                         trips=0, play=10, folded=False)
    assert with_trips == without - 5.0


def test_trips_pays_even_when_player_loses_hand():
    board = cards("7h", "7d", "2c", "9s", "Kd")
    player = cards("7c", "3d")            # trips sevens
    dealer = cards("Kc", "Ks")            # full house kings over sevens
    reward = settle_uth(player, dealer, board, ante=10, blind=10, trips=5,
                        play=10, folded=False)
    # Lose ante+play+blind (-30), Trips pays 3:1 (+15).
    assert reward == -30.0 + 15.0


# ---------------------------------------------------------------------------
# Paytable constants (UTH-01..04)
# ---------------------------------------------------------------------------

def test_paytable_constants():
    for name in ("UTH-01", "UTH-02", "UTH-03", "UTH-04"):
        blind = PAYTABLES[name]["blind"]
        assert blind[HandRank.ROYAL_FLUSH] == 500.0
        assert blind[HandRank.STRAIGHT_FLUSH] == 50.0
        assert blind[HandRank.FOUR_OF_A_KIND] == 10.0
        assert blind[HandRank.FULL_HOUSE] == 3.0
        assert blind[HandRank.FLUSH] == 1.5
        assert blind[HandRank.STRAIGHT] == 1.0
        assert HandRank.THREE_OF_A_KIND not in blind  # push below straight

        trips = PAYTABLES[name]["trips"]
        assert trips[HandRank.ROYAL_FLUSH] == 50.0
        assert trips[HandRank.STRAIGHT_FLUSH] == 40.0
        assert trips[HandRank.THREE_OF_A_KIND] == 3.0

    assert PAYTABLES["UTH-01"]["trips"][HandRank.FOUR_OF_A_KIND] == 30.0
    assert PAYTABLES["UTH-01"]["trips"][HandRank.FULL_HOUSE] == 9.0
    assert PAYTABLES["UTH-01"]["trips"][HandRank.FLUSH] == 7.0
    assert PAYTABLES["UTH-01"]["trips"][HandRank.STRAIGHT] == 4.0

    assert PAYTABLES["UTH-02"]["trips"][HandRank.FULL_HOUSE] == 8.0
    assert PAYTABLES["UTH-02"]["trips"][HandRank.FLUSH] == 6.0
    assert PAYTABLES["UTH-02"]["trips"][HandRank.STRAIGHT] == 5.0

    assert PAYTABLES["UTH-03"]["trips"][HandRank.FULL_HOUSE] == 8.0
    assert PAYTABLES["UTH-03"]["trips"][HandRank.FLUSH] == 7.0
    assert PAYTABLES["UTH-03"]["trips"][HandRank.STRAIGHT] == 4.0

    assert PAYTABLES["UTH-04"]["trips"][HandRank.FOUR_OF_A_KIND] == 20.0
    assert PAYTABLES["UTH-04"]["trips"][HandRank.FULL_HOUSE] == 7.0
    assert PAYTABLES["UTH-04"]["trips"][HandRank.FLUSH] == 6.0
    assert PAYTABLES["UTH-04"]["trips"][HandRank.STRAIGHT] == 5.0


def test_blind_must_equal_ante():
    board = cards("2c", "5d", "9h", "Jc", "Kd")
    with pytest.raises(ValueError):
        settle_uth(cards("As", "Ks"), cards("2s", "3s"), board,
                   ante=10, blind=5, trips=0, play=10)


# ---------------------------------------------------------------------------
# Play wager legality per stage (3x/4x preflop, 2x flop, 1x river)
# ---------------------------------------------------------------------------

def _fresh_state(seed=0):
    return UTHState.from_seed(ante=10.0, bonus_bet=0.0, stake=1000.0, seed=seed)


@pytest.mark.parametrize("mult, ok", [(3.0, True), (4.0, True),
                                      (2.0, False), (1.0, False)])
def test_preflop_play_multipliers(mult, ok):
    state = _fresh_state()
    assert state.stage == UTHStage.Preflop
    new = state.apply_action(UTHAction(UTHActionEnum.Raise, mult))
    if ok:
        assert new.status == UTHStatus.Ok
        assert new.final_state
        assert new.play == mult * 10.0
    else:
        assert new.status != UTHStatus.Ok


@pytest.mark.parametrize("mult, ok", [(2.0, True), (3.0, False),
                                      (4.0, False), (1.0, False)])
def test_flop_play_multipliers(mult, ok):
    state = _fresh_state().apply_action(UTHAction(UTHActionEnum.Check))
    assert state.stage == UTHStage.Flop
    assert len(state.public_cards) == 3
    new = state.apply_action(UTHAction(UTHActionEnum.Raise, mult))
    assert (new.status == UTHStatus.Ok) is ok


@pytest.mark.parametrize("mult, ok", [(1.0, True), (2.0, False),
                                      (3.0, False), (4.0, False)])
def test_river_play_multipliers(mult, ok):
    state = (_fresh_state()
             .apply_action(UTHAction(UTHActionEnum.Check))
             .apply_action(UTHAction(UTHActionEnum.Check)))
    assert state.stage == UTHStage.River
    assert len(state.public_cards) == 5
    new = state.apply_action(UTHAction(UTHActionEnum.Raise, mult))
    assert (new.status == UTHStatus.Ok) is ok


def test_fold_only_legal_at_river():
    preflop = _fresh_state()
    assert UTHActionEnum.Fold not in preflop.legal_actions
    assert preflop.apply_action(UTHAction(UTHActionEnum.Fold)).status != UTHStatus.Ok

    flop = preflop.apply_action(UTHAction(UTHActionEnum.Check))
    assert UTHActionEnum.Fold not in flop.legal_actions

    river = flop.apply_action(UTHAction(UTHActionEnum.Check))
    assert UTHActionEnum.Fold in river.legal_actions
    assert UTHActionEnum.Check not in river.legal_actions

    folded = river.apply_action(UTHAction(UTHActionEnum.Fold))
    assert folded.status == UTHStatus.Ok
    assert folded.final_state
    assert folded.reward == -20.0  # lost ante + blind, no trips


def test_exact_payout_royal_flush_full_ride():
    """Player plays 4x preflop and rivers a royal: exact chip count."""
    player = cards("As", "Ks")
    dealer = cards("2c", "2d")   # qualifies with a pair
    board = cards("Qs", "Js", "Ts", "3h", "4h")
    ante = 10.0
    reward = settle_uth(player, dealer, board, ante=ante, blind=ante,
                        trips=ante, play=4 * ante, folded=False)
    # Ante 1:1 (+10), Play 1:1 (+40), Blind 500:1 (+5000), Trips 50:1 (+500).
    assert reward == 10.0 + 40.0 + 5000.0 + 500.0
