# src/envs/uth_env.py
"""
Ultimate Texas Hold'em (UTH) environment.

NOTE ON THE pokers ENGINE
-------------------------
The Rust `pokers` engine in this project ships a `BonusState` class, but it
implements *Texas Hold'em Bonus Poker* (Fold/Play 2x preflop, 1x flop/turn
bets) - NOT Ultimate Texas Hold'em. UTH needs an Ante+Blind structure,
3x/4x - 2x - 1x Play bets, a Blind paytable and a Trips side bet, none of
which exist in the engine. Following the "prefer a Python-layer wrapper"
guideline, this module implements UTH fully in Python while mirroring the
`BonusState` API style (stage / legal_actions / apply_action / reward /
final_state / status) so agents and training code stay consistent.

Rules implemented (BGC rules sheet):
- Player posts equal Ante and Blind, plus an optional Trips side bet.
- Preflop: Check, or Play bet of 3x or 4x Ante.
- Flop (if checked before): Check, or Play bet of 2x Ante.
- River (if checked before): Fold, or Play bet of 1x Ante.
- Player-dealer qualifies with a Pair or better.
  - Not qualified: Ante pushes; Play/Blind resolved normally.
  - Qualified: win -> Ante & Play pay 1:1; lose -> Ante, Play, Blind lost;
    tie -> push.
- Blind pays per paytable only when the player wins with a Straight or
  better; a win below a Straight pushes the Blind.
- Trips pays per paytable when the player's final hand is Three of a Kind
  or better - even if the player folds; otherwise the Trips bet is lost.
"""

import itertools
import random
from enum import IntEnum

from src.envs.uth_paytables import (
    BLIND_PAYTABLE,
    DEFAULT_PAYTABLE,
    FLUSH,
    HIGH_CARD,
    PAIR,
    ROYAL_FLUSH,
    STRAIGHT,
    STRAIGHT_FLUSH,
    THREE_OF_A_KIND,
    FOUR_OF_A_KIND,
    FULL_HOUSE,
    TWO_PAIR,
    get_trips_paytable,
)

# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

SUITS = ("c", "d", "h", "s")
RANKS = ("2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A")
RANK_VALUES = {r: i + 2 for i, r in enumerate(RANKS)}  # 2..14 (A high)


class Card:
    """Immutable playing card. rank: 2..14 (14 = Ace), suit: 0..3."""

    __slots__ = ("rank", "suit")

    def __init__(self, rank, suit):
        object.__setattr__(self, "rank", rank)
        object.__setattr__(self, "suit", suit)

    def __setattr__(self, *args):
        raise AttributeError("Card is immutable")

    @staticmethod
    def from_string(s):
        """Parse a card like 'As', 'Td', '9c'."""
        if len(s) != 2 or s[0].upper() not in RANK_VALUES or s[1].lower() not in SUITS:
            raise ValueError(f"Invalid card string: {s!r}")
        return Card(RANK_VALUES[s[0].upper()], SUITS.index(s[1].lower()))

    def index(self):
        """0..51 index compatible with the existing one-hot encodings."""
        return self.suit * 13 + (self.rank - 2)

    def __eq__(self, other):
        return isinstance(other, Card) and self.rank == other.rank and self.suit == other.suit

    def __hash__(self):
        return hash((self.rank, self.suit))

    def __repr__(self):
        return f"{RANKS[self.rank - 2]}{SUITS[self.suit]}"


def make_deck():
    return [Card(rank, suit) for suit in range(4) for rank in range(2, 15)]


def cards_from_strings(strings):
    return [Card.from_string(s) for s in strings]


# ---------------------------------------------------------------------------
# Hand evaluation (best 5 of 7)
# ---------------------------------------------------------------------------

def _rank_5(cards):
    """Rank exactly 5 cards. Returns (category, tiebreakers tuple)."""
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    # Straight detection (Ace can play low: A-2-3-4-5)
    unique = sorted(set(ranks), reverse=True)
    straight_high = None
    if len(unique) == 5:
        if unique[0] - unique[4] == 4:
            straight_high = unique[0]
        elif unique == [14, 5, 4, 3, 2]:  # wheel
            straight_high = 5

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Sort by (count, rank) descending -> e.g. full house: [(3, r), (2, r)]
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)
    pattern = tuple(g[1] for g in groups)
    group_ranks = tuple(g[0] for g in groups)

    if is_flush and straight_high is not None:
        if straight_high == 14:
            return (ROYAL_FLUSH, (14,))
        return (STRAIGHT_FLUSH, (straight_high,))
    if pattern == (4, 1):
        return (FOUR_OF_A_KIND, group_ranks)
    if pattern == (3, 2):
        return (FULL_HOUSE, group_ranks)
    if is_flush:
        return (FLUSH, tuple(ranks))
    if straight_high is not None:
        return (STRAIGHT, (straight_high,))
    if pattern == (3, 1, 1):
        return (THREE_OF_A_KIND, group_ranks)
    if pattern == (2, 2, 1):
        return (TWO_PAIR, group_ranks)
    if pattern == (2, 1, 1, 1):
        return (PAIR, group_ranks)
    return (HIGH_CARD, tuple(ranks))


def evaluate_hand(cards):
    """
    Evaluate the best 5-card hand from 5-7 cards.

    Returns (category, tiebreakers) - tuples compare correctly with > / <.
    """
    if len(cards) < 5:
        raise ValueError(f"Need at least 5 cards, got {len(cards)}")
    if len(cards) == 5:
        return _rank_5(cards)
    return max(_rank_5(list(combo)) for combo in itertools.combinations(cards, 5))


# ---------------------------------------------------------------------------
# Actions / statuses / stages
# ---------------------------------------------------------------------------

class UTHActionEnum(IntEnum):
    Fold = 0     # River only
    Check = 1    # Preflop / Flop
    Bet1x = 2    # River only
    Bet2x = 3    # Flop only
    Bet3x = 4    # Preflop only
    Bet4x = 5    # Preflop only


# Play-bet multiple committed by each action
ACTION_MULTIPLES = {
    UTHActionEnum.Bet1x: 1.0,
    UTHActionEnum.Bet2x: 2.0,
    UTHActionEnum.Bet3x: 3.0,
    UTHActionEnum.Bet4x: 4.0,
}

NUM_UTH_ACTIONS = len(UTHActionEnum)


class UTHStatus(IntEnum):
    Ok = 0
    IllegalAction = 1


class UTHStage(IntEnum):
    Preflop = 0
    Flop = 1
    River = 2
    Showdown = 3


STAGE_LEGAL_ACTIONS = {
    UTHStage.Preflop: [UTHActionEnum.Check, UTHActionEnum.Bet3x, UTHActionEnum.Bet4x],
    UTHStage.Flop: [UTHActionEnum.Check, UTHActionEnum.Bet2x],
    UTHStage.River: [UTHActionEnum.Fold, UTHActionEnum.Bet1x],
    UTHStage.Showdown: [],
}


# ---------------------------------------------------------------------------
# State
# ---------------------------------------------------------------------------

class UTHState:
    """
    Immutable-style UTH state; `apply_action` returns a new state.

    Mirrors the pokers `BonusState` API: stage, player_hand, dealer_hand,
    public_cards, deck, ante, blind, trips_bet, play_bet, stake, reward,
    legal_actions, final_state, status, from_action, dealer_revealed.
    """

    def __init__(self):
        self.stage = UTHStage.Preflop
        self.player_hand = ()
        self.dealer_hand = ()
        self.public_cards = []
        self.deck = []
        self.ante = 0.0
        self.blind = 0.0
        self.trips_bet = 0.0
        self.play_bet = 0.0
        self.stake = 0.0
        self.reward = 0.0
        self.legal_actions = list(STAGE_LEGAL_ACTIONS[UTHStage.Preflop])
        self.final_state = False
        self.status = UTHStatus.Ok
        self.from_action = None
        self.dealer_revealed = False
        self.folded = False
        self.paytable = DEFAULT_PAYTABLE
        # Filled at settlement for inspection / testing
        self.info = {}

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_seed(ante, blind=None, trips_bet=0.0, stake=1000.0, seed=0,
                  paytable=DEFAULT_PAYTABLE):
        """
        Start a hand from a seeded shuffle.

        Blind must equal Ante per the rules; defaults to ante when omitted.
        """
        if blind is None:
            blind = ante
        if abs(blind - ante) > 1e-9:
            raise ValueError("UTH requires the Blind wager to equal the Ante.")
        if ante <= 0:
            raise ValueError("Ante must be positive.")
        if trips_bet < 0:
            raise ValueError("Trips bet cannot be negative.")
        deck = make_deck()
        random.Random(seed).shuffle(deck)
        return UTHState._from_deck_order(deck, ante, blind, trips_bet, stake, paytable)

    @staticmethod
    def from_deck(deck, ante, blind=None, trips_bet=0.0, stake=1000.0,
                  paytable=DEFAULT_PAYTABLE):
        """
        Start a hand from an explicit deck order (deterministic, for tests).

        Deal order: player 2 cards, dealer 2 cards, then board 5 as needed.
        """
        if blind is None:
            blind = ante
        if len(deck) < 9:
            raise ValueError("Deck must contain at least 9 cards.")
        if len(set(deck)) != len(deck):
            raise ValueError("Deck contains duplicate cards.")
        return UTHState._from_deck_order(list(deck), ante, blind, trips_bet, stake, paytable)

    @staticmethod
    def _from_deck_order(deck, ante, blind, trips_bet, stake, paytable=DEFAULT_PAYTABLE):
        get_trips_paytable(paytable)  # validate early
        state = UTHState()
        state.ante = float(ante)
        state.blind = float(blind)
        state.trips_bet = float(trips_bet)
        state.stake = float(stake)
        state.paytable = paytable
        deck = list(deck)
        state.player_hand = (deck.pop(0), deck.pop(0))
        state.dealer_hand = (deck.pop(0), deck.pop(0))
        state.deck = deck
        return state

    # -- helpers ------------------------------------------------------------

    def _clone(self):
        new = UTHState.__new__(UTHState)
        new.stage = self.stage
        new.player_hand = self.player_hand
        new.dealer_hand = self.dealer_hand
        new.public_cards = list(self.public_cards)
        new.deck = list(self.deck)
        new.ante = self.ante
        new.blind = self.blind
        new.trips_bet = self.trips_bet
        new.play_bet = self.play_bet
        new.stake = self.stake
        new.reward = self.reward
        new.legal_actions = list(self.legal_actions)
        new.final_state = self.final_state
        new.status = self.status
        new.from_action = self.from_action
        new.dealer_revealed = self.dealer_revealed
        new.folded = self.folded
        new.paytable = self.paytable
        new.info = dict(self.info)
        return new

    def _deal_board_to(self, n_cards):
        while len(self.public_cards) < n_cards:
            self.public_cards.append(self.deck.pop(0))

    # -- core transition ----------------------------------------------------

    def apply_action(self, action):
        """Apply a UTHActionEnum; returns a new state (defensive on illegal)."""
        new = self._clone()
        new.from_action = action

        if self.final_state or action not in self.legal_actions:
            new.status = UTHStatus.IllegalAction
            return new

        new.status = UTHStatus.Ok

        if action == UTHActionEnum.Fold:
            # River only: lose Ante + Blind; Trips still resolves on the
            # player's final 7-card hand (all 5 board cards are already out).
            new._deal_board_to(5)
            new.folded = True
            new._settle()
            return new

        if action in ACTION_MULTIPLES:
            new.play_bet = ACTION_MULTIPLES[action] * new.ante
            new._deal_board_to(5)
            new._settle()
            return new

        # Check: advance to the next decision point
        if self.stage == UTHStage.Preflop:
            new.stage = UTHStage.Flop
            new._deal_board_to(3)
        elif self.stage == UTHStage.Flop:
            new.stage = UTHStage.River
            new._deal_board_to(5)
        new.legal_actions = list(STAGE_LEGAL_ACTIONS[new.stage])
        return new

    # -- settlement ---------------------------------------------------------

    def _settle(self):
        self.stage = UTHStage.Showdown
        self.legal_actions = []
        self.final_state = True
        self.dealer_revealed = True

        player_eval = evaluate_hand(list(self.player_hand) + self.public_cards)
        dealer_eval = evaluate_hand(list(self.dealer_hand) + self.public_cards)
        player_cat = player_eval[0]
        dealer_qualifies = dealer_eval[0] >= PAIR

        ante_net = 0.0
        blind_net = 0.0
        play_net = 0.0

        if self.folded:
            ante_net = -self.ante
            blind_net = -self.blind
        else:
            player_wins = player_eval > dealer_eval
            dealer_wins = dealer_eval > player_eval

            # Ante: pushes when the player-dealer does not qualify
            if dealer_qualifies:
                if player_wins:
                    ante_net = self.ante
                elif dealer_wins:
                    ante_net = -self.ante
            # Play: always resolved normally
            if player_wins:
                play_net = self.play_bet
            elif dealer_wins:
                play_net = -self.play_bet
            # Blind: pays only on a win with Straight or better
            if player_wins:
                if player_cat in BLIND_PAYTABLE:
                    blind_net = BLIND_PAYTABLE[player_cat] * self.blind
                # else: push (0)
            elif dealer_wins:
                blind_net = -self.blind

        # Trips: independent of the base game; pays even on a fold
        trips_net = 0.0
        if self.trips_bet > 0:
            trips_table = get_trips_paytable(self.paytable)
            if player_cat in trips_table:
                trips_net = trips_table[player_cat] * self.trips_bet
            else:
                trips_net = -self.trips_bet

        self.reward = ante_net + blind_net + play_net + trips_net
        self.stake += self.reward
        self.info = {
            "player_eval": player_eval,
            "dealer_eval": dealer_eval,
            "dealer_qualifies": dealer_qualifies,
            "folded": self.folded,
            "ante_net": ante_net,
            "blind_net": blind_net,
            "play_net": play_net,
            "trips_net": trips_net,
        }

    def __str__(self):
        return (
            f"UTHState(stage={self.stage.name}, hand={self.player_hand}, "
            f"board={self.public_cards}, play_bet={self.play_bet}, "
            f"final={self.final_state}, reward={self.reward})"
        )


# ---------------------------------------------------------------------------
# Gym-style environment wrapper
# ---------------------------------------------------------------------------

class UTHEnv:
    """
    Gym-style wrapper around UTHState.

    reset(seed) -> state
    step(action) -> (next_state, reward, done, info)
    legal_actions(state) -> list[UTHActionEnum]
    """

    def __init__(self, ante=10.0, trips_bet=0.0, stake=1000.0,
                 paytable=DEFAULT_PAYTABLE):
        get_trips_paytable(paytable)  # validate early
        self.ante = ante
        self.trips_bet = trips_bet
        self.stake = stake
        self.paytable = paytable
        self.state = None

    def reset(self, seed=0):
        self.state = UTHState.from_seed(
            ante=self.ante,
            trips_bet=self.trips_bet,
            stake=self.stake,
            seed=seed,
            paytable=self.paytable,
        )
        return self.state

    @staticmethod
    def legal_actions(state):
        return list(state.legal_actions)

    def step(self, action):
        if self.state is None:
            raise RuntimeError("Call reset() before step().")
        new_state = self.state.apply_action(action)
        if new_state.status != UTHStatus.Ok:
            raise ValueError(
                f"Illegal action {action!r} at stage {self.state.stage.name}; "
                f"legal: {self.state.legal_actions}"
            )
        self.state = new_state
        reward = new_state.reward if new_state.final_state else 0.0
        return new_state, reward, new_state.final_state, dict(new_state.info)
