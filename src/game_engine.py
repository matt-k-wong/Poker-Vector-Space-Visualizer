"""
Game engine module for poker game logic and state management.
"""

import random
from pokerkit import NoLimitTexasHoldem, Automation, Deck, Card
from typing import Tuple, List, Optional, Dict, Any


class PokerGame:
    """Manages the poker game state and logic."""

    def __init__(
        self, player_stacks: List[int] = [100, 100], blinds: List[int] = [1, 2]
    ):
        """
        Initialize a new poker game.

        Args:
            player_stacks: Starting chip stacks for each player
            blinds: [small_blind, big_blind] amounts
        """
        self.player_stacks = player_stacks
        self.blinds = blinds
        self.reset_game()

    def reset_game(self) -> None:
        """Reset the game to initial state."""
        self.game_state = NoLimitTexasHoldem.create_state(
            (
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
            ),
            False,  # uniform_antes
            [],  # antes
            self.blinds,  # blinds
            100,  # min_bet
            self.player_stacks,  # stacks
            len(self.player_stacks),  # num_players
        )
        self.hand_over = False

    def is_human_turn(self, human_player_index: int = 0) -> bool:
        """
        Check if it's the human player's turn.

        Args:
            human_player_index: Index of the human player (default 0)

        Returns:
            True if it's human player's turn
        """
        return (
            self.game_state.status and self.game_state.actor_index == human_player_index
        )

    def is_game_over(self) -> bool:
        """Check if the current hand is over."""
        return not self.game_state.status

    def get_current_player(self) -> int:
        """Get the index of the player whose turn it is."""
        return (
            self.game_state.actor_index
            if self.game_state.actor_index is not None
            else 0
        )

    def get_hole_cards(self, player_index: int) -> List:
        """Get hole cards for a player."""
        if (
            self.game_state.hole_cards
            and len(self.game_state.hole_cards) > player_index
            and self.game_state.hole_cards[player_index]
        ):
            return self.game_state.hole_cards[player_index]
        return []

    def get_board_cards(self) -> List:
        """Get community cards on the board."""
        if self.game_state.board_cards:
            return self.game_state.board_cards[0]
        return []

    def get_pot_amount(self) -> int:
        """Get current pot amount."""
        return self.game_state.total_pot_amount

    def get_player_stack(self, player_index: int) -> int:
        """Get chip stack for a player."""
        return self.game_state.stacks[player_index]

    def can_check_or_call(self) -> bool:
        """Check if current player can check or call."""
        return self.game_state.can_check_or_call()

    def can_fold(self) -> bool:
        """Check if current player can fold."""
        return self.game_state.can_fold()

    def can_bet_or_raise(self) -> bool:
        """Check if current player can bet or raise."""
        return self.game_state.can_complete_bet_or_raise_to()

    def get_min_bet(self) -> int:
        """Get minimum bet/raise amount."""
        return self.game_state.min_completion_betting_or_raising_to_amount

    def get_max_bet(self) -> int:
        """Get maximum bet/raise amount."""
        return self.game_state.max_completion_betting_or_raising_to_amount

    def check_or_call(self) -> None:
        """Execute check or call action."""
        self.game_state.check_or_call()

    def fold(self) -> None:
        """Execute fold action."""
        self.game_state.fold()

    def bet_or_raise(self, amount: int) -> None:
        """Execute bet or raise action."""
        self.game_state.complete_bet_or_raise_to(amount)

    def advance_game(self) -> None:
        """Advance game with simple AI for non-human players."""
        if not self.game_state.status:
            return

        current_player = self.get_current_player()
        # Simple AI: only acts if it's not human player's turn
        if current_player != 0:  # Assuming human is player 0
            if self.can_check_or_call():
                self.check_or_call()
            elif self.can_fold():
                self.fold()

    def get_hand_result(self, player_index: int = 0) -> Dict[str, Any]:
        """
        Get result information for completed hand.

        Args:
            player_index: Index of player to get results for

        Returns:
            Dictionary with hand result information
        """
        payout = (
            self.get_player_stack(player_index) - 100
        )  # Assuming 100 starting stack
        return {
            "payout": payout,
            "player_stack": self.get_player_stack(player_index),
            "pot_amount": self.get_pot_amount(),
            "hole_cards": self.get_hole_cards(player_index),
            "board_cards": self.get_board_cards(),
        }


def create_sample_hands() -> Tuple[List[str], List[str]]:
    """
    Create sample hands for demonstration purposes.

    Returns:
        Tuple of (sample_hands, sample_actions)
    """
    sample_hands = [
        "AA",
        "KK",
        "QQ",
        "JJ",
        "AK suited",
        "AQ suited",
        "KQ suited",
        "77",
        "66",
        "A9 offsuit",
        "K8 suited",
        "Q10 offsuit",
        "J9 suited",
    ]
    sample_actions = ["Check/Call", "Bet/Raise", "Fold", "Check/Call", "Bet/Raise"]
    return sample_hands, sample_actions
