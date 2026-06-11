# src/agents/uth_dealer_agent.py
"""
Fixed-rule dealer for Ultimate Texas Hold'em.

In UTH the dealer (player-dealer) makes no decisions: their hand is simply
revealed at showdown and they "play with a pair or higher" (qualification).
This agent exists to make that explicit in evaluation code and to give tests
a single place to query dealer-side rules.
"""

from src.core.uth_env import HandRank, dealer_qualifies, evaluate_seven


class UTHDealerAgent:
    """Rules-based, non-learnable dealer. Never chooses an action."""

    def __init__(self, player_id=1):
        self.player_id = player_id
        self.name = f"UTHDealerAgent_{player_id}"

    def choose_action(self, state):
        """The dealer never acts in UTH; chance auto-advances in UTHState."""
        raise RuntimeError(
            "UTH dealer has no decisions; UTHState.apply_action auto-advances."
        )

    @staticmethod
    def qualifies(dealer_hand, board):
        """Dealer plays with a pair or higher."""
        return dealer_qualifies(dealer_hand, board)

    @staticmethod
    def best_hand(dealer_hand, board):
        """The dealer's best 5-of-7 hand value."""
        return evaluate_seven(list(dealer_hand) + list(board))

    @staticmethod
    def qualification_threshold():
        return HandRank.PAIR
