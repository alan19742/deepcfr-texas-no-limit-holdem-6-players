# src/core/uth_env.py
"""
Ultimate Texas Hold'em (UTH) environment.

The `pokers` fork only ships `BonusState` (Texas Hold'em *Bonus* Poker), which
is a different casino game. This module implements the real UTH rules (BGC,
revised March 2015) in pure Python while staying duck-type compatible with the
`pokers.State` interface so `src/core/model.py::encode_state` works unchanged.

Rules implemented:
  - Ante and Blind are mandatory, equal wagers. Trips is an optional side bet.
  - The player makes exactly one Play wager at one of three decision points:
      Preflop : Check, or Play 3x / 4x Ante
      Flop    : Check, or Play 2x Ante
      River   : Fold, or Play 1x Ante
  - The dealer qualifies with a pair or better. If the dealer does NOT
    qualify, the Ante pushes (it is NOT an automatic win); Play / Blind /
    Trips are still resolved normally.
  - Ante / Play pay 1:1 on a player win, push on tie, lose on dealer win.
  - Blind pays per paytable only when the player wins with a straight or
    better; a win below a straight pushes the Blind; a loss loses the Blind.
  - Trips pays per paytable whenever the player's final 7-card hand makes
    three of a kind or better - independent of win/lose/fold.

Card representation matches `pokers`: suit in 0..3, rank in 0..12 (0=2,...,
12=Ace), `card_idx = suit * 13 + rank`.
"""

import copy
import random
from enum import IntEnum
from itertools import combinations

VERBOSE = False


def set_verbose(verbose_mode):
    """Set the module verbosity level (mirrors src/core/model.py)."""
    global VERBOSE
    VERBOSE = verbose_mode


# ---------------------------------------------------------------------------
# Cards
# ---------------------------------------------------------------------------

class UTHCard:
    """Minimal card compatible with encode_state (int(card.suit/rank))."""

    __slots__ = ("suit", "rank")

    def __init__(self, suit, rank):
        self.suit = int(suit)
        self.rank = int(rank)

    def __eq__(self, other):
        return self.suit == other.suit and self.rank == other.rank

    def __hash__(self):
        return hash((self.suit, self.rank))

    def __repr__(self):
        ranks = "23456789TJQKA"
        suits = "cdhs"
        return f"{ranks[self.rank]}{suits[self.suit]}"


_RANK_CHARS = {c: i for i, c in enumerate("23456789TJQKA")}
_SUIT_CHARS = {"c": 0, "d": 1, "h": 2, "s": 3}


def card(text):
    """Parse a card like 'As', 'Td', '9c' into a UTHCard."""
    rank_char, suit_char = text[0].upper(), text[1].lower()
    return UTHCard(_SUIT_CHARS[suit_char], _RANK_CHARS[rank_char])


def full_deck():
    """Ordered 52-card deck, same indexing convention as pokers."""
    return [UTHCard(s, r) for s in range(4) for r in range(13)]


# ---------------------------------------------------------------------------
# Hand evaluation (7 choose 5, higher tuple = better hand)
# ---------------------------------------------------------------------------

class HandRank(IntEnum):
    HIGH_CARD = 0
    PAIR = 1
    TWO_PAIR = 2
    THREE_OF_A_KIND = 3
    STRAIGHT = 4
    FLUSH = 5
    FULL_HOUSE = 6
    FOUR_OF_A_KIND = 7
    STRAIGHT_FLUSH = 8
    ROYAL_FLUSH = 9


def _straight_high(ranks_desc_unique):
    """Return the high card of a straight in the given unique ranks, or None.
    Handles the wheel (A-2-3-4-5) where the high card is 5 (rank index 3)."""
    rs = set(ranks_desc_unique)
    # Ace can play low
    if 12 in rs:
        rs.add(-1)
    best = None
    for high in range(12, 2, -1):
        if all((high - i) in rs for i in range(5)):
            best = high
            break
    return best


def evaluate_five(cards):
    """Evaluate exactly 5 cards. Returns (HandRank, tiebreak tuple)."""
    ranks = sorted((c.rank for c in cards), reverse=True)
    suits = [c.suit for c in cards]
    is_flush = len(set(suits)) == 1

    counts = {}
    for r in ranks:
        counts[r] = counts.get(r, 0) + 1
    # Sort by count desc then rank desc
    groups = sorted(counts.items(), key=lambda kv: (kv[1], kv[0]), reverse=True)

    straight_high = _straight_high(sorted(set(ranks), reverse=True))

    if is_flush and straight_high is not None:
        if straight_high == 12:
            return (HandRank.ROYAL_FLUSH, (12,))
        return (HandRank.STRAIGHT_FLUSH, (straight_high,))

    if groups[0][1] == 4:
        quad = groups[0][0]
        kicker = max(r for r in ranks if r != quad)
        return (HandRank.FOUR_OF_A_KIND, (quad, kicker))

    if groups[0][1] == 3 and groups[1][1] >= 2:
        return (HandRank.FULL_HOUSE, (groups[0][0], groups[1][0]))

    if is_flush:
        return (HandRank.FLUSH, tuple(ranks))

    if straight_high is not None:
        return (HandRank.STRAIGHT, (straight_high,))

    if groups[0][1] == 3:
        trip = groups[0][0]
        kickers = tuple(r for r in ranks if r != trip)
        return (HandRank.THREE_OF_A_KIND, (trip,) + kickers)

    if groups[0][1] == 2 and groups[1][1] == 2:
        hi_pair, lo_pair = groups[0][0], groups[1][0]
        kicker = max(r for r in ranks if r != hi_pair and r != lo_pair)
        return (HandRank.TWO_PAIR, (hi_pair, lo_pair, kicker))

    if groups[0][1] == 2:
        pair = groups[0][0]
        kickers = tuple(r for r in ranks if r != pair)
        return (HandRank.PAIR, (pair,) + kickers)

    return (HandRank.HIGH_CARD, tuple(ranks))


def evaluate_seven(cards):
    """Best 5-card evaluation among 7 cards. Returns (HandRank, tiebreaks)."""
    best = None
    for combo in combinations(cards, 5):
        val = evaluate_five(combo)
        if best is None or val > best:
            best = val
    return best


# ---------------------------------------------------------------------------
# Paytables (BGC, revised March 2015)
# ---------------------------------------------------------------------------

# Blind paytable is identical across UTH-01..04. Hands below straight push.
_BLIND_TABLE = {
    HandRank.ROYAL_FLUSH: 500.0,
    HandRank.STRAIGHT_FLUSH: 50.0,
    HandRank.FOUR_OF_A_KIND: 10.0,
    HandRank.FULL_HOUSE: 3.0,
    HandRank.FLUSH: 1.5,
    HandRank.STRAIGHT: 1.0,
}

def _trips_table(quads, full_house, flush, straight):
    return {
        HandRank.ROYAL_FLUSH: 50.0,
        HandRank.STRAIGHT_FLUSH: 40.0,
        HandRank.FOUR_OF_A_KIND: quads,
        HandRank.FULL_HOUSE: full_house,
        HandRank.FLUSH: flush,
        HandRank.STRAIGHT: straight,
        HandRank.THREE_OF_A_KIND: 3.0,
    }

PAYTABLES = {
    "UTH-01": {"blind": _BLIND_TABLE, "trips": _trips_table(30.0, 9.0, 7.0, 4.0)},
    "UTH-02": {"blind": _BLIND_TABLE, "trips": _trips_table(30.0, 8.0, 6.0, 5.0)},
    "UTH-03": {"blind": _BLIND_TABLE, "trips": _trips_table(30.0, 8.0, 7.0, 4.0)},
    "UTH-04": {"blind": _BLIND_TABLE, "trips": _trips_table(20.0, 7.0, 6.0, 5.0)},
}

DEFAULT_PAYTABLE = "UTH-01"


# ---------------------------------------------------------------------------
# Settlement (pure function, unit-testable)
# ---------------------------------------------------------------------------

def settle_uth(player_hand, dealer_hand, board, ante, blind, trips, play,
               folded=False, paytable=DEFAULT_PAYTABLE):
    """
    Compute the player's net profit/loss for one finished UTH round.

    Args:
        player_hand: list/tuple of 2 UTHCard
        dealer_hand: list/tuple of 2 UTHCard
        board: list of 5 UTHCard (full board; UTH folds only happen on the
               river, after all 5 community cards are out)
        ante / blind / trips / play: wager amounts (blind must equal ante;
               play is 0.0 when folded)
        folded: True if the player folded at the river decision
        paytable: one of PAYTABLES keys

    Returns:
        Net reward (float). Trips is always resolved on the player's best
        7-card hand, even on a fold.
    """
    if blind != ante:
        raise ValueError("Blind wager must equal the Ante wager")
    tables = PAYTABLES[paytable]

    player_best = evaluate_seven(list(player_hand) + list(board))
    reward = 0.0

    # Trips side bet: independent of fold / win / lose / push.
    if trips > 0:
        mult = tables["trips"].get(player_best[0])
        reward += trips * mult if mult is not None else -trips

    if folded:
        # Fold loses Ante and Blind (Play was never made).
        return reward - ante - blind

    dealer_best = evaluate_seven(list(dealer_hand) + list(board))
    dealer_qualifies = dealer_best[0] >= HandRank.PAIR

    player_wins = player_best > dealer_best
    dealer_wins = dealer_best > player_best

    # Ante: pushes when the dealer does not qualify (NOT an automatic win).
    if dealer_qualifies:
        if player_wins:
            reward += ante
        elif dealer_wins:
            reward -= ante

    # Play: resolved normally regardless of dealer qualification.
    if player_wins:
        reward += play
    elif dealer_wins:
        reward -= play

    # Blind: paytable on a win with straight or better, push on a lesser win.
    if player_wins:
        mult = tables["blind"].get(player_best[0])
        if mult is not None:
            reward += blind * mult
        # else: push
    elif dealer_wins:
        reward -= blind

    return reward


def dealer_qualifies(dealer_hand, board):
    """True when the dealer's best 7-card hand is a pair or better."""
    return evaluate_seven(list(dealer_hand) + list(board))[0] >= HandRank.PAIR


# ---------------------------------------------------------------------------
# State machine (duck-type compatible with pokers.State for encode_state)
# ---------------------------------------------------------------------------

class UTHActionEnum(IntEnum):
    """Mirrors pokers.ActionEnum integer layout (Fold/Check/Call/Raise)."""
    Fold = 0
    Check = 1
    Call = 2   # unused in UTH; kept for encoding compatibility
    Raise = 3  # "Play" wager; amount = multiplier of the ante


class UTHStatus(IntEnum):
    Ok = 0
    IllegalAction = 1


class UTHStage(IntEnum):
    """Matches pokers.Stage indices used by encode_state's 5-dim one-hot."""
    Preflop = 0
    Flop = 1
    Turn = 2      # never a decision point in UTH
    River = 3
    Showdown = 4


# Legal Play multipliers per decision stage.
PLAY_MULTIPLIERS = {
    UTHStage.Preflop: (3.0, 4.0),
    UTHStage.Flop: (2.0,),
    UTHStage.River: (1.0,),
}


class UTHAction:
    """Player action; for Raise (Play) `amount` is the ante multiplier."""

    __slots__ = ("action", "amount")

    def __init__(self, action, amount=0.0):
        self.action = action
        self.amount = float(amount)

    def __repr__(self):
        if self.action == UTHActionEnum.Raise:
            return f"UTHAction(Play x{self.amount:g})"
        return f"UTHAction({self.action.name})"


class UTHActionRecord:
    """Mimics pokers.ActionRecord (`state.from_action.action.{action,amount}`)."""

    __slots__ = ("player", "action")

    def __init__(self, player, action):
        self.player = player
        self.action = action


class UTHPlayerState:
    """Mimics pokers.PlayerState for encode_state."""

    __slots__ = ("player", "hand", "bet_chips", "pot_chips", "stake",
                 "reward", "active")

    def __init__(self, player, hand, stake):
        self.player = player
        self.hand = hand
        self.bet_chips = 0.0
        self.pot_chips = 0.0
        self.stake = stake
        self.reward = 0.0
        self.active = True


class UTHState:
    """
    Single-player-vs-dealer UTH state. Player is players_state[0]; the dealer
    (players_state[1]) never acts - chance events auto-advance inside
    apply_action, so current_player is always 0 until the hand is final.
    """

    def __init__(self):
        # Filled by from_deck.
        self.stage = UTHStage.Preflop
        self.players_state = []
        self.public_cards = []
        self.deck = []
        self.board = []           # full pre-dealt board (hidden until revealed)
        self.ante = 0.0
        self.blind = 0.0
        self.trips = 0.0
        self.play = 0.0
        self.paytable = DEFAULT_PAYTABLE
        self.reward = 0.0
        self.legal_actions = []
        self.final_state = False
        self.status = UTHStatus.Ok
        self.from_action = None
        self.current_player = 0
        self.button = 0
        self.min_bet = 0.0
        self.folded = False

    # -- constructors -------------------------------------------------------

    @staticmethod
    def from_seed(ante, bonus_bet, stake, seed, paytable=DEFAULT_PAYTABLE):
        """Signature aligned with pkrs.BonusState.from_seed; `bonus_bet` is
        the Trips side bet."""
        rng = random.Random(seed)
        deck = full_deck()
        rng.shuffle(deck)
        return UTHState.from_deck(ante, bonus_bet, stake, deck, paytable)

    @staticmethod
    def from_deck(ante, bonus_bet, stake, deck, paytable=DEFAULT_PAYTABLE):
        """Deal order: player x2, dealer x2, board x5 (top of deck first)."""
        if ante <= 0.0:
            raise ValueError("ante must be greater than 0")
        if bonus_bet < 0.0:
            raise ValueError("bonus_bet (Trips) must be non-negative")
        # Worst case commitment: ante + blind + trips + 4x ante play.
        if stake < ante * 2 + bonus_bet + 4.0 * ante:
            raise ValueError("stake must cover ante + blind + trips + 4 * ante")
        if len(deck) < 9:
            raise ValueError("deck must contain at least 9 cards")
        if paytable not in PAYTABLES:
            raise ValueError(f"unknown paytable {paytable!r}")

        deck = list(deck)
        s = UTHState()
        s.ante = float(ante)
        s.blind = float(ante)
        s.trips = float(bonus_bet)
        s.paytable = paytable

        player_hand = (deck.pop(0), deck.pop(0))
        dealer_hand = (deck.pop(0), deck.pop(0))
        s.board = [deck.pop(0) for _ in range(5)]
        s.deck = deck

        committed = s.ante + s.blind + s.trips
        player = UTHPlayerState(0, player_hand, float(stake) - committed)
        player.bet_chips = committed
        dealer = UTHPlayerState(1, dealer_hand, float(stake))
        s.players_state = [player, dealer]

        s.pot = committed
        s.legal_actions = s._compute_legal_actions()
        return s

    # -- helpers ------------------------------------------------------------

    def _compute_legal_actions(self):
        if self.final_state:
            return []
        if self.play > 0:
            return []
        if self.stage == UTHStage.Preflop:
            return [UTHActionEnum.Check, UTHActionEnum.Raise]
        if self.stage == UTHStage.Flop:
            return [UTHActionEnum.Check, UTHActionEnum.Raise]
        if self.stage == UTHStage.River:
            return [UTHActionEnum.Fold, UTHActionEnum.Raise]
        return []

    def legal_play_multipliers(self):
        """Legal Play-bet multipliers at the current decision point."""
        return PLAY_MULTIPLIERS.get(self.stage, ())

    def _clone(self):
        new = UTHState.__new__(UTHState)
        new.stage = self.stage
        new.players_state = copy.deepcopy(self.players_state)
        new.public_cards = list(self.public_cards)
        new.deck = list(self.deck)
        new.board = list(self.board)
        new.ante = self.ante
        new.blind = self.blind
        new.trips = self.trips
        new.play = self.play
        new.paytable = self.paytable
        new.reward = self.reward
        new.legal_actions = list(self.legal_actions)
        new.final_state = self.final_state
        new.status = self.status
        new.from_action = self.from_action
        new.current_player = self.current_player
        new.button = self.button
        new.min_bet = self.min_bet
        new.pot = self.pot
        new.folded = self.folded
        return new

    def _reveal(self, upto):
        self.public_cards = list(self.board[:upto])

    def _commit_play(self, multiplier):
        self.play = multiplier * self.ante
        player = self.players_state[0]
        player.stake -= self.play
        player.bet_chips += self.play
        self.pot += self.play

    def _settle(self):
        self.stage = UTHStage.Showdown
        self._reveal(5)
        reward = settle_uth(
            self.players_state[0].hand,
            self.players_state[1].hand,
            self.board,
            self.ante, self.blind, self.trips, self.play,
            folded=self.folded,
            paytable=self.paytable,
        )
        self.reward = reward
        self.players_state[0].reward = reward
        self.players_state[1].reward = -reward
        self.final_state = True
        self.legal_actions = []

    # -- transitions ---------------------------------------------------------

    def apply_action(self, action):
        """Apply a UTHAction and return the new state (immutable transition).
        Illegal actions return a state with status != Ok (pokers-style)."""
        if self.final_state or self.status != UTHStatus.Ok:
            return self._clone()

        s = self._clone()
        s.from_action = UTHActionRecord(0, action)

        action_enum = action.action

        if action_enum not in self.legal_actions:
            s.status = UTHStatus.IllegalAction
            s.final_state = True
            s.legal_actions = []
            return s

        if action_enum == UTHActionEnum.Raise:
            if action.amount not in self.legal_play_multipliers():
                s.status = UTHStatus.IllegalAction
                s.final_state = True
                s.legal_actions = []
                return s
            s._commit_play(action.amount)
            s._settle()
            return s

        if action_enum == UTHActionEnum.Fold:
            # Only legal at the river.
            s.folded = True
            player = s.players_state[0]
            player.active = False
            s._settle()
            return s

        # Check: advance to the next decision point (chance auto-advances).
        if self.stage == UTHStage.Preflop:
            s.stage = UTHStage.Flop
            s._reveal(3)
        elif self.stage == UTHStage.Flop:
            s.stage = UTHStage.River
            s._reveal(5)
        s.legal_actions = s._compute_legal_actions()
        return s

    def __str__(self):
        return (f"UTHState(stage={self.stage.name}, board={self.public_cards}, "
                f"ante={self.ante}, play={self.play}, trips={self.trips}, "
                f"final={self.final_state}, reward={self.reward})")
