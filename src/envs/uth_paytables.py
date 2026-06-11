# src/envs/uth_paytables.py
"""
Ultimate Texas Hold'em paytables (BGC rules, options UTH-01 .. UTH-04).

Hand category constants follow ascending strength. Payouts are expressed as
"X to 1" multipliers (a 3:2 payout is 1.5).

Source: BGC Ultimate Texas Hold'em rules sheet, section 12.
- The Blind paytable is identical across all four options.
- The Trips paytables differ per option.
"""

# Hand categories (ascending strength)
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

HAND_CATEGORY_NAMES = {
    HIGH_CARD: "High Card",
    PAIR: "Pair",
    TWO_PAIR: "Two Pair",
    THREE_OF_A_KIND: "Three of a Kind",
    STRAIGHT: "Straight",
    FLUSH: "Flush",
    FULL_HOUSE: "Full House",
    FOUR_OF_A_KIND: "Four of a Kind",
    STRAIGHT_FLUSH: "Straight Flush",
    ROYAL_FLUSH: "Royal Flush",
}

# Blind wager paytable ("X to 1"). Pays only when the player WINS the hand
# with a Straight or better; a winning hand below a Straight pushes the Blind.
# Identical for UTH-01 .. UTH-04.
BLIND_PAYTABLE = {
    ROYAL_FLUSH: 500.0,
    STRAIGHT_FLUSH: 50.0,
    FOUR_OF_A_KIND: 10.0,
    FULL_HOUSE: 3.0,
    FLUSH: 1.5,  # 3 to 2
    STRAIGHT: 1.0,
}

# Trips bonus paytables ("X to 1"). Pays whenever the player's final 7-card
# hand makes Three of a Kind or better - even if the player folds.
TRIPS_PAYTABLES = {
    "UTH-01": {
        ROYAL_FLUSH: 50.0,
        STRAIGHT_FLUSH: 40.0,
        FOUR_OF_A_KIND: 30.0,
        FULL_HOUSE: 9.0,
        FLUSH: 7.0,
        STRAIGHT: 4.0,
        THREE_OF_A_KIND: 3.0,
    },
    "UTH-02": {
        ROYAL_FLUSH: 50.0,
        STRAIGHT_FLUSH: 40.0,
        FOUR_OF_A_KIND: 30.0,
        FULL_HOUSE: 8.0,
        FLUSH: 6.0,
        STRAIGHT: 5.0,
        THREE_OF_A_KIND: 3.0,
    },
    "UTH-03": {
        ROYAL_FLUSH: 50.0,
        STRAIGHT_FLUSH: 40.0,
        FOUR_OF_A_KIND: 30.0,
        FULL_HOUSE: 8.0,
        FLUSH: 7.0,
        STRAIGHT: 4.0,
        THREE_OF_A_KIND: 3.0,
    },
    "UTH-04": {
        ROYAL_FLUSH: 50.0,
        STRAIGHT_FLUSH: 40.0,
        FOUR_OF_A_KIND: 20.0,
        FULL_HOUSE: 7.0,
        FLUSH: 6.0,
        STRAIGHT: 5.0,
        THREE_OF_A_KIND: 3.0,
    },
}

DEFAULT_PAYTABLE = "UTH-01"


def get_trips_paytable(name=DEFAULT_PAYTABLE):
    """Return the Trips paytable dict for a given option name (e.g. 'UTH-01')."""
    if name not in TRIPS_PAYTABLES:
        raise ValueError(
            f"Unknown paytable '{name}'. Valid options: {sorted(TRIPS_PAYTABLES)}"
        )
    return TRIPS_PAYTABLES[name]
