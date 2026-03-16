"""
Vector analysis module for poker state vectorization and EV calculations.
"""

import numpy as np
from typing import Tuple, Optional, List
from pokerkit import Card, Hand, Deck, NoLimitTexasHoldem, Automation


def vectorize_state(
    hole_cards: List,
    board_cards: List,
    bet_amount: float = 0,
    pot_amount: float = 0,
    player_position: int = 0,
    num_players: int = 2,
) -> np.ndarray:
    """
    Convert poker game state to a feature vector.

    Args:
        hole_cards: List of hole card objects
        board_cards: List of board card objects
        bet_amount: Current bet amount for the player
        pot_amount: Current pot amount
        player_position: Position of the player (0-indexed)
        num_players: Total number of players

    Returns:
        114-dimensional feature vector
    """
    vector = np.zeros(114)

    # Hole cards encoding (52 dimensions)
    if hole_cards:
        for card in hole_cards:
            # Calculate card ID from rank and suit
            # pokerkit rank.value returns string like 'A', 'K', 'Q', etc.
            # suit.value returns string like 'c', 'd', 'h', 's'
            rank_value = card.rank.value  # String like 'A', 'K', 'Q', '2', etc.
            suit_value = card.suit.value  # String like 'c', 'd', 'h', 's'

            # Map rank string to numeric value
            rank_map = {
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
                "T": 10,
                "J": 11,
                "Q": 12,
                "K": 13,
                "A": 14,
            }
            rank_numeric = rank_map[rank_value]

            # Map suit string to numeric index
            suit_map = {"c": 0, "d": 1, "h": 2, "s": 3}  # CLUB, DIAMOND, HEART, SPADE
            suit_index = suit_map[suit_value]

            # Convert rank to 0-12 index (2->0, 3->1, ..., 14->12)
            rank_index = rank_numeric - 2
            # Suit is already 0-3 (CLUB=0, DIAMOND=1, HEART=2, SPADE=3)
            # Calculate unique card ID: 0-51
            card_id = rank_index * 4 + suit_index
            if 0 <= card_id < 52:
                vector[card_id] = 1

    # Board cards encoding (52 dimensions)
    if board_cards:
        for card in board_cards:
            # Calculate card ID from rank and suit
            rank_value = card.rank.value  # String like 'A', 'K', 'Q', etc.
            suit_value = card.suit.value  # String like 'c', 'd', 'h', 's'

            # Map rank string to numeric value
            rank_map = {
                "2": 2,
                "3": 3,
                "4": 4,
                "5": 5,
                "6": 6,
                "7": 7,
                "8": 8,
                "9": 9,
                "T": 10,
                "J": 11,
                "Q": 12,
                "K": 13,
                "A": 14,
            }
            rank_numeric = rank_map[rank_value]

            # Map suit string to numeric index
            suit_map = {"c": 0, "d": 1, "h": 2, "s": 3}  # CLUB, DIAMOND, HEART, SPADE
            suit_index = suit_map[suit_value]

            # Convert rank to 0-12 index (2->0, 3->1, ..., 14->12)
            rank_index = rank_numeric - 2
            # Suit is already 0-3 (CLUB=0, DIAMOND=1, HEART=2, SPADE=3)
            # Calculate unique card ID: 0-51
            card_id = rank_index * 4 + suit_index
            if 0 <= card_id < 52:
                vector[52 + card_id] = 1

    # Betting features (4 dimensions)
    if pot_amount > 0:
        vector[104] = min(
            bet_amount / pot_amount, 1.0
        )  # Normalized bet size (capped at 1.0)
    else:
        vector[104] = 0

    # Position features (2 dimensions)
    vector[105 + player_position] = 1

    # Game state features (4 dimensions - could be extended)
    vector[107] = min(bet_amount / 200.0, 1.0)  # Normalized bet size (cap at 200)
    vector[108] = len(board_cards) / 5.0  # Progress through hand (0 to 1)
    vector[109] = player_position / max(num_players - 1, 1)  # Normalized position
    vector[110] = (num_players - 2) / 2.0  # Number of opponents normalized

    return vector


def estimate_ev_and_guide(
    hole_cards: List,
    board_cards: List,
    bet_amount: float = 0,
    pot_amount: float = 0,
    action: Optional[str] = None,
    chosen_bet_amount: Optional[float] = None,
) -> Tuple[float, float, str]:
    """
    Estimate expected value and provide optimal play guidance.

    Args:
        hole_cards: List of hole card objects (pokerkit Card objects)
        board_cards: List of board card objects (pokerkit Card objects)
        bet_amount: Current bet amount to call
        pot_amount: Current pot amount
        action: Action taken ('fold', 'call', 'bet')
        chosen_bet_amount: Amount bet/raised if action was bet

    Returns:
        Tuple of (ev_penalty, perfection_score, guidance_text)
    """
    if not hole_cards or len(hole_cards) != 2:
        return 0.0, 100.0, "No valid hand."

    try:
        # Calculate hand equity using pokerkit
        if board_cards:
            # Create a temporary hand to evaluate
            temp_hand = NoLimitTexasHoldem.create_state(
                (
                    Automation.ANTE_POSTING,
                    Automation.BET_COLLECTION,
                    Automation.BLIND_OR_STRADDLE_POSTING,
                    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                    Automation.HAND_KILLING,
                    Automation.CHIPS_PUSHING,
                    Automation.CHIPS_PULLING,
                ),
                False,  # uniform_antes
                [],  # antes
                [],  # blinds
                0,  # min_bet
                [100, 100],  # stacks
                2,  # num_players
            )

            # Set the cards for evaluation
            temp_hand._state["hole_cards"] = [tuple(hole_cards)]
            temp_hand._state["board_cards"] = [tuple(board_cards)]

            # Get hand rank and description
            hand_rank, hand_type_str = temp_hand.evaluate_hand()

            # Convert rank to equity (this is a simplification)
            # In a real implementation, we would use precomputed equity tables
            # or run simulations. For now, we'll use a heuristic based on hand strength.
            equity = hand_rank_to_equity(hand_rank)
        else:
            # Preflop equity estimation (simplified)
            # Convert pokerkit cards to strings for our existing function
            hole_card_strings = [str(card) for card in hole_cards]
            treys_hole = [
                Card.from_str(cs) for cs in hole_card_strings
            ]  # Using pokerkit's Card
            equity = estimate_preflop_equity(treys_hole)

        # Calculate pot odds
        pot_odds = (
            bet_amount / (pot_amount + bet_amount)
            if (pot_amount + bet_amount) > 0
            else 0
        )

        # Determine optimal action based on equity and pot odds
        optimal_action = "fold"
        optimal_bet = 0

        if equity > 0.7:  # Strong hand
            optimal_action = "bet"
            optimal_bet = pot_amount * 0.5  # Half pot bet
        elif equity > pot_odds and pot_amount > 0:  # Pot odds justify call
            optimal_action = "call"
        # Otherwise, fold is optimal

        # Calculate EVs
        optimal_ev = 0
        if optimal_action == "bet":
            optimal_ev = equity * (pot_amount + optimal_bet) - optimal_bet
        elif optimal_action == "call":
            optimal_ev = equity * pot_amount - bet_amount
        # fold EV is 0

        # Calculate actual EV based on player's action
        actual_ev = 0
        if action == "bet" and chosen_bet_amount:
            actual_ev = equity * (pot_amount + chosen_bet_amount) - chosen_bet_amount
        elif action == "call":
            actual_ev = equity * pot_amount - bet_amount
        elif action == "fold":
            actual_ev = 0
        # If no action taken yet, actual EV is 0

        # Calculate EV penalty (how much worse than optimal)
        ev_penalty = actual_ev - optimal_ev

        # Convert to perfection score (0-100, where 100 is perfect)
        # Scale factor of 50 means 2bb penalty = 0 score
        perfection_score = max(0, 100 - abs(ev_penalty) * 50)

        # Generate guidance text
        action_text = ""
        if optimal_action == "bet":
            action_text = f"{optimal_action} {optimal_bet:.0f}"
        else:
            action_text = optimal_action

        guide = f"Optimal: {action_text}. "

        if action and perfection_score < 90:
            guide += f"Your {action} loses {ev_penalty:.2f} bb vs. optimal."
        elif action and perfection_score >= 90:
            guide += "Nailed it!"
        else:
            guide += "Make your decision."

        return ev_penalty, perfection_score, guide
    except Exception as e:
        # Fallback if evaluation fails
        return 0.0, 100.0, f"Evaluation error: {str(e)}"


def hand_rank_to_equity(hand_rank: int) -> float:
    """
    Convert pokerkit hand rank to equity estimate.
    This is a simplified mapping - in production, use precomputed equity tables.

    Args:
        hand_rank: Pokerkit hand rank (lower is better)

    Returns:
        Estimated equity (0-1)
    """
    # Pokerkit returns ranks where lower is better
    # We'll map the rank to a 0-1 equity score
    # This is a simplification - real equity depends on board texture, etc.
    if hand_rank >= 9000:  # Very weak hands
        return 0.05
    elif hand_rank >= 8000:  # Weak hands
        return 0.15
    elif hand_rank >= 7000:  # Below average hands
        return 0.25
    elif hand_rank >= 6000:  # Average hands
        return 0.35
    elif hand_rank >= 5000:  # Above average hands
        return 0.45
    elif hand_rank >= 4000:  # Good hands
        return 0.55
    elif hand_rank >= 3000:  # Very good hands
        return 0.65
    elif hand_rank >= 2000:  # Excellent hands
        return 0.75
    elif hand_rank >= 1000:  # Premium hands
        return 0.85
    else:  # Monster hands
        return 0.95


def estimate_preflop_equity(hole_cards: List) -> float:
    """
    Estimate preflop equity for a hole card pair.
    Uses pokerkit card objects directly.

    Args:
        hole_cards: List of two pokerkit Card objects

    Returns:
        Estimated equity against random hand (0-1)
    """
    if len(hole_cards) != 2:
        return 0.5

    # Extract ranks using pokerkit's rank property
    rank1 = hole_cards[0].rank
    rank2 = hole_cards[1].rank
    suited = hole_cards[0].suit == hole_cards[1].suit  # Same suit

    # Convert rank to numeric value (2-14, where J=11, Q=12, K=13, A=14)
    # pokerkit ranks are already numeric: 2-10, J=11, Q=12, K=13, A=14
    r1 = rank1
    r2 = rank2

    # Preflop equity estimation (simplified)
    if r1 == r2:  # Pocket pair
        # Pair equity increases with rank
        base_equity = 0.5 + (r1 - 7) * 0.03  # 22=0.5, AA=0.86
        return min(max(base_equity, 0.1), 0.9)
    elif suited:  # Suited cards
        # Suited connectivity and high card strength
        high_card = max(r1, r2)
        gap = abs(r1 - r2)

        base_equity = 0.35 + (high_card - 7) * 0.04  # Base from high card
        if gap == 1:  # Connected
            base_equity += 0.1
        elif gap == 2:  # One-gap
            base_equity += 0.05
        # Suited bonus
        base_equity += 0.05
        return min(max(base_equity, 0.1), 0.9)
    else:  # Offsuit cards
        high_card = max(r1, r2)
        gap = abs(r1 - r2)

        base_equity = 0.3 + (high_card - 7) * 0.035  # Base from high card
        if gap == 0:  # Pair (shouldn't reach here due to earlier check)
            base_equity += 0.1
        elif gap == 1:  # Connected
            base_equity += 0.05
        elif gap == 2:  # One-gap
            base_equity += 0.02
        return min(max(base_equity, 0.1), 0.9)


def calculate_hand_strength(hole_cards: List, board_cards: List) -> Tuple[int, str]:
    """
    Calculate actual hand strength and description using pokerkit.

    Args:
        hole_cards: List of hole card objects (pokerkit Card objects)
        board_cards: List of board card objects (pokerkit Card objects)

    Returns:
        Tuple of (hand_rank, hand_description)
    """
    if not hole_cards or len(hole_cards) != 2:
        return 9999, "Invalid hand"

    try:
        # Create a temporary hand state for evaluation
        temp_hand = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            False,  # uniform_antes
            [],  # antes
            [],  # blinds
            0,  # min_bet
            [100, 100],  # stacks
            2,  # num_players
        )

        # Set the cards for evaluation
        if hole_cards:
            temp_hand._state["hole_cards"] = [tuple(hole_cards)]
        if board_cards:
            temp_hand._state["board_cards"] = [tuple(board_cards)]

        # Get hand rank and description
        hand_rank, hand_type_str = temp_hand.evaluate_hand()

        return hand_rank, hand_type_str
    except Exception:
        return 9999, "Error evaluating hand"
