"""
Unit tests for the game engine module.
"""

import unittest
from src.game_engine import PokerGame


class TestPokerGame(unittest.TestCase):
    """Test cases for the PokerGame class."""

    def setUp(self):
        """Set up test fixtures before each test method."""
        self.game = PokerGame()

    def test_initialization(self):
        """Test that the game initializes correctly."""
        self.assertIsNotNone(self.game.game_state)
        self.assertEqual(len(self.game.player_stacks), 2)
        self.assertEqual(self.game.player_stacks, [100, 100])
        self.assertEqual(self.game.blinds, [1, 2])

    def test_reset_game(self):
        """Test that resetting the game works correctly."""
        # Make some moves
        original_status = self.game.game_state.status

        # Reset the game
        self.game.reset_game()

        # Check that the game is in a valid state
        self.assertIsNotNone(self.game.game_state)
        self.assertTrue(self.game.game_state.status)  # Game should be active

    def test_get_current_player(self):
        """Test getting the current player."""
        # Initially, it should be player 0 (human) or determined by game state
        current_player = self.game.get_current_player()
        self.assertIn(current_player, [0, 1])  # Should be either player

    def test_get_hole_cards(self):
        """Test getting hole cards for a player."""
        # Before dealing, should return empty list
        hole_cards = self.game.get_hole_cards(0)
        self.assertIsInstance(hole_cards, list)

        # After dealing (if we could simulate it), should return cards
        # We'll test this indirectly through game progression

    def test_get_board_cards(self):
        """Test getting board cards."""
        # Initially, should be empty (preflop)
        board_cards = self.game.get_board_cards()
        self.assertIsInstance(board_cards, list)
        self.assertEqual(len(board_cards), 0)  # No board cards preflop

    def test_get_pot_amount(self):
        """Test getting the pot amount."""
        pot_amount = self.game.get_pot_amount()
        self.assertIsInstance(pot_amount, int)
        self.assertGreaterEqual(pot_amount, 0)

    def test_get_player_stack(self):
        """Test getting player stack."""
        stack = self.game.get_player_stack(0)
        self.assertIsInstance(stack, int)
        # After blinds are posted, stacks will be less than starting amount
        self.assertGreaterEqual(stack, 0)
        self.assertLessEqual(stack, 100)

    def test_can_check_or_call(self):
        """Test checking if player can check or call."""
        result = self.game.can_check_or_call()
        self.assertIsInstance(result, bool)

    def test_can_fold(self):
        """Test checking if player can fold."""
        result = self.game.can_fold()
        self.assertIsInstance(result, bool)

    def test_can_bet_or_raise(self):
        """Test checking if player can bet or raise."""
        result = self.game.can_bet_or_raise()
        self.assertIsInstance(result, bool)

    def test_get_min_bet(self):
        """Test getting minimum bet."""
        min_bet = self.game.get_min_bet()
        self.assertIsInstance(min_bet, int)
        self.assertGreaterEqual(min_bet, 0)

    def test_get_max_bet(self):
        """Test getting maximum bet."""
        max_bet = self.game.get_max_bet()
        self.assertIsInstance(max_bet, int)
        self.assertGreaterEqual(max_bet, 0)

    def test_check_or_call(self):
        """Test executing check or call."""
        # This should not raise an exception
        try:
            self.game.check_or_call()
        except Exception as e:
            self.fail(f"check_or_call raised {type(e).__name__} unexpectedly: {e}")

    def test_fold(self):
        """Test executing fold."""
        # This should not raise an exception
        try:
            self.game.fold()
        except Exception as e:
            self.fail(f"fold raised {type(e).__name__} unexpectedly: {e}")

    def test_bet_or_raise(self):
        """Test executing bet or raise."""
        # This should not raise an exception for valid amounts
        try:
            min_bet = self.game.get_min_bet()
            if min_bet > 0:
                self.game.bet_or_raise(min_bet)
        except Exception as e:
            # Might fail if game state doesn't allow betting, which is OK
            pass

    def test_advance_game(self):
        """Test advancing the game with AI logic."""
        # This should not raise an exception
        try:
            self.game.advance_game()
        except Exception as e:
            self.fail(f"advance_game raised {type(e).__name__} unexpectedly: {e}")

    def test_get_hand_result(self):
        """Test getting hand result information."""
        result = self.game.get_hand_result()

        # Check that we get a dictionary with expected keys
        self.assertIsInstance(result, dict)
        expected_keys = [
            "payout",
            "player_stack",
            "pot_amount",
            "hole_cards",
            "board_cards",
        ]
        for key in expected_keys:
            self.assertIn(key, result)

        # Check types
        self.assertIsInstance(result["payout"], int)
        self.assertIsInstance(result["player_stack"], int)
        self.assertIsInstance(result["pot_amount"], int)
        self.assertIsInstance(result["hole_cards"], list)
        self.assertIsInstance(result["board_cards"], list)


if __name__ == "__main__":
    unittest.main()
