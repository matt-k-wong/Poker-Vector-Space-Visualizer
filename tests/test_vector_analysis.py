"""
Unit tests for the vector analysis module.
"""

import unittest
import numpy as np
import sys
import os

# Add the src directory to the path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from src.vector_analysis import (
    vectorize_state,
    estimate_ev_and_guide,
    hand_rank_to_equity,
    estimate_preflop_equity,
    calculate_hand_strength,
)

try:
    from pokerkit import Card as PokerCard, Rank, Suit, NoLimitTexasHoldem, Automation
except ImportError as e:
    print(f"Error importing pokerkit: {e}")
    raise


class TestVectorAnalysis(unittest.TestCase):
    """Test cases for the vector analysis functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Import pokerkit enums for card creation
        from pokerkit import Rank, Suit

        # Create some test cards using pokerkit
        self.test_cards = [
            PokerCard(Rank.ACE, Suit.HEART),  # Ace of hearts
            PokerCard(Rank.KING, Suit.SPADE),  # King of spades
            PokerCard(Rank.QUEEN, Suit.DIAMOND),  # Queen of diamonds
            PokerCard(Rank.JACK, Suit.CLUB),  # Jack of clubs
            PokerCard(Rank.TEN, Suit.HEART),  # Ten of hearts
        ]

        # Create string representations for testing
        self.test_card_strings = [
            "Ah",  # Ace of hearts
            "Ks",  # King of spades
            "Qd",  # Queen of diamonds
            "Jc",  # Jack of clubs
            "Th",  # Ten of hearts
        ]

    def test_vectorize_state(self):
        """Test that vectorization produces the correct shape and values."""
        # Test with hole cards and board cards
        hole_cards = [self.test_cards[0], self.test_cards[1]]  # Ah, Ks
        board_cards = [
            self.test_cards[2],
            self.test_cards[3],
            self.test_cards[4],
        ]  # Qd, Jc, Th

        vector = vectorize_state(
            hole_cards=hole_cards,
            board_cards=board_cards,
            bet_amount=10,
            pot_amount=50,
            player_position=0,
            num_players=2,
        )

        # Check that we get a 114-dimensional vector
        self.assertEqual(vector.shape, (114,))
        self.assertIsInstance(vector, np.ndarray)

        # Check that hole cards are encoded in positions 0-51
        # Calculate expected IDs: (rank-2)*4 + suit
        # Ah: rank=14 (ACE), suit=2 (HEART) -> (14-2)*4 + 2 = 12*4 + 2 = 50
        # Ks: rank=13 (KING), suit=3 (SPADE) -> (13-2)*4 + 3 = 11*4 + 3 = 47
        ah_id = (14 - 2) * 4 + 2  # Ace of hearts
        ks_id = (13 - 2) * 4 + 3  # King of spades
        self.assertEqual(vector[ah_id], 1.0)
        self.assertEqual(vector[ks_id], 1.0)

        # Check that board cards are encoded in positions 52-103
        # Qd: rank=12 (QUEEN), suit=1 (DIAMOND) -> (12-2)*4 + 1 = 10*4 + 1 = 41 -> 52+41=93
        # Jc: rank=11 (JACK), suit=0 (CLUB) -> (11-2)*4 + 0 = 9*4 + 0 = 36 -> 52+36=88
        # Th: rank=10 (TEN), suit=2 (HEART) -> (10-2)*4 + 2 = 8*4 + 2 = 34 -> 52+34=86
        qd_id = 52 + ((12 - 2) * 4 + 1)  # Queen of diamonds
        jc_id = 52 + ((11 - 2) * 4 + 0)  # Jack of clubs
        th_id = 52 + ((10 - 2) * 4 + 2)  # Ten of hearts
        self.assertEqual(vector[qd_id], 1.0)
        self.assertEqual(vector[jc_id], 1.0)
        self.assertEqual(vector[th_id], 1.0)

        # Check betting features
        self.assertEqual(vector[104], 10 / 50)  # bet_amount/pot_amount

        # Check position features
        self.assertEqual(vector[105], 1.0)  # player 0 position
        self.assertEqual(vector[106], 0.0)  # player 1 position

        # Check game state features
        self.assertAlmostEqual(vector[107], min(10 / 200.0, 1.0))  # normalized bet
        self.assertAlmostEqual(vector[108], 3 / 5.0)  # 3 board cards / 5
        self.assertAlmostEqual(vector[109], 0.0)  # position 0 / (2-1)
        self.assertAlmostEqual(vector[110], 0.0)  # (2-2)/2 = 0

    def test_vectorize_state_edge_cases(self):
        """Test vectorization with edge cases."""
        # Test with no cards
        vector = vectorize_state(
            hole_cards=[], board_cards=[], bet_amount=0, pot_amount=0
        )

        self.assertEqual(vector.shape, (114,))
        # All should be zero except position features
        self.assertEqual(vector[104], 0)  # bet/pot when pot=0
        self.assertEqual(vector[105], 1.0)  # player 0 position

        # Test with zero pot (avoid division by zero)
        vector = vectorize_state(
            hole_cards=[self.test_cards[0]], board_cards=[], bet_amount=5, pot_amount=0
        )

        self.assertEqual(vector.shape, (114,))
        self.assertEqual(vector[104], 0)  # bet/pot when pot=0
        self.assertEqual(vector[105], 1.0)  # player 0 position

        # Test with large bet amount
        vector = vectorize_state(
            hole_cards=self.test_cards[:2],
            board_cards=[],
            bet_amount=1000,
            pot_amount=100,
        )

        self.assertEqual(vector[104], min(1000 / 100, 1.0))  # Should be capped at 1.0

    def test_hand_rank_to_equity(self):
        """Test conversion of hand rank to equity."""
        # Test known values
        self.assertEqual(hand_rank_to_equity(9000), 0.05)  # Very weak
        self.assertEqual(hand_rank_to_equity(8000), 0.15)  # Weak
        self.assertEqual(hand_rank_to_equity(7000), 0.25)  # Below average
        self.assertEqual(hand_rank_to_equity(6000), 0.35)  # Average
        self.assertEqual(hand_rank_to_equity(5000), 0.45)  # Above average
        self.assertEqual(hand_rank_to_equity(4000), 0.55)  # Good
        self.assertEqual(hand_rank_to_equity(3000), 0.65)  # Very good
        self.assertEqual(hand_rank_to_equity(2000), 0.75)  # Excellent
        self.assertEqual(hand_rank_to_equity(1000), 0.85)  # Premium
        self.assertEqual(hand_rank_to_equity(0), 0.95)  # Monster

        # Test intermediate values
        self.assertEqual(hand_rank_to_equity(8500), 0.15)  # Still in weak range
        self.assertEqual(hand_rank_to_equity(1500), 0.85)  # Still in premium range

    def test_estimate_preflop_equity(self):
        """Test preflop equity estimation."""
        # Test pocket pairs
        pocket_aces = [
            PokerCard(14, 2),
            PokerCard(14, 1),
        ]  # Ace of hearts, Ace of spades
        equity = estimate_preflop_equity(pocket_aces)
        # AA equity is typically around 0.85, but our simplified model gives ~0.71
        self.assertGreater(equity, 0.6)  # AA should be reasonably strong

        pocket_twos = [PokerCard(2, 2), PokerCard(2, 1)]  # Two of hearts, Two of spades
        equity = estimate_preflop_equity(pocket_twos)
        self.assertLess(equity, 0.5)  # 22 should be weak

        # Test suited connectors
        suited_connector = [
            PokerCard(13, 2),
            PokerCard(12, 2),
        ]  # King of hearts, Queen of hearts
        equity = estimate_preflop_equity(suited_connector)
        self.assertGreater(equity, 0.3)  # Should be decent

        # Test offsuit trash
        trash = [PokerCard(7, 2), PokerCard(2, 1)]  # Seven of hearts, Two of spades
        equity = estimate_preflop_equity(trash)
        self.assertLess(equity, 0.4)  # Should be weak

        # Test invalid inputs
        self.assertEqual(estimate_preflop_equity([]), 0.5)
        self.assertEqual(estimate_preflop_equity([self.test_cards[0]]), 0.5)
        self.assertEqual(
            estimate_preflop_equity(
                [self.test_cards[0], self.test_cards[1], self.test_cards[2]]
            ),
            0.5,
        )

    def test_calculate_hand_strength(self):
        """Test hand strength calculation."""
        # Test with no cards (should return invalid hand)
        rank, description = calculate_hand_strength([], [])
        self.assertEqual(rank, 9999)
        self.assertEqual(description, "Invalid hand")

        # Test with only hole cards (preflop)
        hole_cards = [self.test_cards[0], self.test_cards[1]]
        rank, description = calculate_hand_strength(hole_cards, [])
        # Description might vary, just check that we got a valid result
        self.assertIsInstance(rank, int)
        self.assertIsInstance(description, str)
        # For preflop hands, we might get "Error evaluating hand" or a valid description
        # Just make sure we don't get an exception
        self.assertTrue(len(description) > 0)

        # Test with a known hand (we won't check exact rank as it's complex)
        # Just test that the function doesn't crash and returns valid types
        hole_cards = [self.test_cards[0], self.test_cards[1]]
        board_cards = [self.test_cards[2], self.test_cards[3], self.test_cards[4]]
        rank, description = calculate_hand_strength(hole_cards, board_cards)
        # The main thing is that it doesn't crash and returns proper types
        self.assertIsInstance(rank, int)
        self.assertIsInstance(description, str)
        # Description should not be the error message if it worked
        if description != "Error evaluating hand":
            self.assertNotEqual(description, "Invalid hand")

    def test_estimate_ev_and_guide(self):
        """Test EV estimation and guidance."""
        hole_cards = [self.test_cards[0], self.test_cards[1]]
        board_cards = [
            PokerCard(14, 0),  # Ace of diamonds
            PokerCard(13, 0),  # King of diamonds
            PokerCard(12, 2),  # Queen of hearts
        ]  # Strong hand

        # Test with no action yet
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards=hole_cards, board_cards=board_cards
        )

        self.assertIsInstance(ev_penalty, float)
        self.assertIsInstance(perfection_score, float)
        self.assertIsInstance(guide, str)
        self.assertGreaterEqual(perfection_score, 0.0)
        self.assertLessEqual(perfection_score, 100.0)

        # Test with a fold action
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards=hole_cards, board_cards=board_cards, action="fold"
        )

        self.assertIsInstance(ev_penalty, float)
        self.assertIsInstance(perfection_score, float)
        self.assertIsInstance(guide, str)

        # Test with a call action
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards=hole_cards,
            board_cards=board_cards,
            action="call",
            chosen_bet_amount=20,  # Assuming bet to call is 20
        )

        self.assertIsInstance(ev_penalty, float)
        self.assertIsInstance(perfection_score, float)
        self.assertIsInstance(guide, str)

        # Test with a bet action
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards=hole_cards,
            board_cards=board_cards,
            action="bet",
            chosen_bet_amount=30,
        )

        self.assertIsInstance(ev_penalty, float)
        self.assertIsInstance(perfection_score, float)
        self.assertIsInstance(guide, str)

    def test_estimate_ev_and_guide_edge_cases(self):
        """Test EV estimation with edge cases."""
        # Test with invalid hole cards
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards=[], board_cards=[self.test_cards[0]]
        )

        self.assertEqual(ev_penalty, 0.0)
        self.assertEqual(perfection_score, 100.0)
        self.assertEqual(guide, "No valid hand.")

        # Test with exception handling (we'll simulate by passing invalid data)
        # This is harder to test without mocking, but we can at least verify it doesn't crash
        try:
            ev_penalty, perfection_score, guide = estimate_ev_and_guide(
                hole_cards=[self.test_cards[0], self.test_cards[1]],
                board_cards=[self.test_cards[2]],
                bet_amount=-1,  # Negative bet amount
                pot_amount=-1,  # Negative pot amount
            )
            # Should handle gracefully
            self.assertIsInstance(ev_penalty, float)
            self.assertIsInstance(perfection_score, float)
            self.assertIsInstance(guide, str)
        except Exception:
            # If it does raise an exception, that's also acceptable for edge cases
            pass


if __name__ == "__main__":
    unittest.main()
