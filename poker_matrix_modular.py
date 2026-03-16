"""
Modular Poker Matrix: Cyberpunk Poker Trainer
Refactored version with persistent storage and performance optimizations.
"""

import streamlit as st
import numpy as np
import pandas as pd
import random
import uuid
from typing import Optional, Tuple, List

# Import our custom modules
from src.game_engine import PokerGame, create_sample_hands
from src.vector_analysis import (
    vectorize_state,
    estimate_ev_and_guide,
    estimate_preflop_equity,
    calculate_hand_strength,
)
from src.visualization import (
    create_umap_reducer,
    prepare_dataframe_for_plotting,
    apply_umap_transformation,
    create_3d_scatter_plot,
    display_hand_history,
    render_game_state_info,
    render_action_buttons,
    render_perfect_guide,
    render_hand_over_screen,
)
from src.database import get_database
from src.visualization import (
    create_umap_reducer,
    prepare_dataframe_for_plotting,
    apply_umap_transformation,
    create_3d_scatter_plot,
    display_hand_history,
    render_game_state_info,
    render_action_buttons,
    render_perfect_guide,
    render_hand_over_screen,
)


# Initialize session state
def initialize_session_state():
    """Initialize session state variables."""
    if "game_engine" not in st.session_state:
        st.session_state.game_engine = PokerGame()

    if "session_id" not in st.session_state:
        st.session_state.session_id = str(uuid.uuid4())

    if "hands" not in st.session_state:
        st.session_state.hands = []
        # Precomputed sample hands for demo (will be replaced with real hands)
        sample_hands, sample_actions = create_sample_hands()
        perfect_vectors = np.random.rand(10, 114)

        for i, v in enumerate(perfect_vectors):
            st.session_state.hands.append(
                {
                    "vector": v,
                    "ev_penalty": np.random.uniform(-0.5, 0.5),
                    "payout": np.random.randint(-50, 100),
                    "hand_type": sample_hands[i % len(sample_hands)],
                    "cluster": "Demo Data",
                    "hole_cards": sample_hands[i % len(sample_hands)],
                    "board_cards": "Sample",
                    "action": sample_actions[i % len(sample_actions)],
                    "perfection_score": np.random.uniform(70, 100),
                    "street": "Demo",
                }
            )

    if "hands_since_last_umap_fit" not in st.session_state:
        st.session_state.hands_since_last_umap_fit = 0

    if "umap_needs_refit" not in st.session_state:
        st.session_state.umap_needs_refit = False

    if "umap_reducer" not in st.session_state:
        st.session_state.umap_reducer = create_umap_reducer()

    if "umap_is_fitted" not in st.session_state:
        st.session_state.umap_is_fitted = False

    # Initialize database
    if "db" not in st.session_state:
        st.session_state.db = get_database()


def main():
    """Main application function."""
    initialize_session_state()

    # Page configuration
    st.set_page_config(
        page_title="Poker Matrix: Cyberpunk Poker Trainer",
        page_icon="🃏",
        layout="wide",
    )

    # Header
    st.set_page_config(
        page_title="Poker Vector Space Visualizer",
        page_icon="🃏",
        layout="wide",
    )

    # Header
    st.title("Poker Vector Space Visualizer")
    st.markdown(
        "**Play No-Limit Hold'em, optimize with a perfect hand guide, and explore your strategy in a neon 3D vector space!**"
    )

    # Instructions expander
    with st.expander("📖 How to Play & Interpret", expanded=False):
        st.markdown("""
        ### 🎯 Game Objective
        Play No-Limit Texas Hold'em against a simple AI opponent. Your goal is to make optimal decisions that maximize your expected value (EV) over time.
        
        ### 🎮 Controls
        - **Check/Call**: Pass or match the current bet
        - **Bet/Raise**: Increase the pot by specifying an amount
        - **Fold**: Give up your hand and forfeit any chance to win the pot
        - **New Hand**: Start a completely new hand
        - **Reset Analytics**: Clear your hand history and restart the 3D analysis
        
        ### 📱 Perfect Hand Guide
        After each action, you'll see:
        - **Perfection Score**: How close your decision was to optimal (0-100%)
        - **EV Penalty**: Expected value lost compared to optimal play (in big blinds)
        - **Tip**: Specific feedback on your decision
        
        ### 📊 3D Strategy Matrix Interpretation
        The neon 3D plot visualizes your poker strategy in vector space:
        
        **Axes (X, Y, Z)**: 
        - These are UMAP-reduced dimensions from your 114-dimensional poker state vectors
        - Similar strategies cluster together in vector space
        
        **Color**: 
        - Red = Negative EV penalty (bad decisions)
        - Green = Positive EV penalty (good decisions)
        - The color bar shows the exact EV penalty in big blinds
        
        **Size**: 
        - Represents the magnitude of your payout (win/loss) for that hand
        - Larger dots = bigger wins/losses
        
        **Symbols**: 
        - Different symbols represent different action clusters (check/call, bet/raise, fold, etc.)
        
        **Hover Over Dots**: 
        - See detailed information about each hand: hole cards, board cards, action taken, hand type, perfection score, and more
        
        ### 📋 Hand History Table
        Shows your most recent hands with:
        - Hole cards and board cards
        - Action taken
        - Hand type classification
        - Perfection score (%)
        - EV penalty (in bb)
        - Payout (chips won/lost)
        
        ### 💡 Tips for Improvement
        1. Aim for high perfection scores (90+)
        2. Notice patterns in the 3D vector space - where do your good/bad decisions cluster?
        3. Use the perfect hand guide to learn optimal strategies
        4. Review your hand history to identify leaks in your game
        """)

    # Get current game state
    game_engine = st.session_state.game_engine
    player_index = 0  # Human player is always index 0

    # Display game state info
    render_game_state_info(game_engine, player_index)

    # Handle player actions
    action_taken = None
    bet_amount = None

    if (
        game_engine.game_state.status
        and game_engine.get_current_player() == player_index
    ):
        action_taken, bet_amount = render_action_buttons(game_engine, player_index)

    # Process action if taken
    if action_taken:
        # Get current game state for analysis
        hole_cards = game_engine.get_hole_cards(player_index)
        board_cards = game_engine.get_board_cards()
        pot_amount = game_engine.get_pot_amount()
        player_bet = (
            game_engine.game_state.bets[player_index]
            if hasattr(game_engine.game_state, "bets")
            else 0
        )

        # Calculate EV and guidance
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards, board_cards, player_bet, pot_amount, action_taken, bet_amount
        )

        # Get actual hand information
        hole_cards_str = (
            ", ".join(str(c) for c in hole_cards) if hole_cards else "Unknown"
        )
        board_cards_str = (
            ", ".join(str(c) for c in board_cards) if board_cards else "Preflop"
        )

        # Determine actual hand type
        hand_type = "Unknown"
        if hole_cards and len(hole_cards) == 2:
            c1, c2 = hole_cards
            if hasattr(c1, "rank") and hasattr(c2, "rank"):
                if str(c1)[0] == str(c2)[0]:  # Pair
                    hand_type = f"Pocket {str(c1)[0]}s"
                elif str(c1)[1] == str(c2)[1]:  # Suited
                    hand_type = f"{str(c1)[0]}{str(c2)[0]} suited"
                else:  # Offsuit
                    hand_type = f"{str(c1)[0]}{str(c2)[0]} offsuit"

        # Add hand to history and database
        hand_data = {
            "vector": vectorize_state(hole_cards, board_cards, player_bet, pot_amount),
            "ev_penalty": ev_penalty,
            "payout": game_engine.get_player_stack(player_index) - 100,
            "hand_type": hand_type,
            "cluster": "Your Hand",
            "hole_cards": hole_cards_str,
            "board_cards": board_cards_str,
            "action": action_taken,
            "perfection_score": perfection_score,
            "street": "Preflop" if not board_cards else f"{len(board_cards)} cards",
        }

        # Add to session state for immediate use
        st.session_state.hands.append(hand_data)

        # Save to database
        st.session_state.db.save_hand(hand_data, st.session_state.session_id)

        # Increment counter for UMAP refitting
        st.session_state.hands_since_last_umap_fit += 1

        # Check if we need to refit UMAP (every 20 hands)
        if st.session_state.hands_since_last_umap_fit >= 20:
            st.session_state.umap_needs_refit = True

        # Show perfect hand guide
        render_perfect_guide(ev_penalty, perfection_score, guide)

    # AI moves - only act if it's the AI's turn (player 1)
    if (
        game_engine.game_state.status and game_engine.get_current_player() == 1
    ):  # AI is player 1
        # Simple AI logic - just check/call or fold
        if game_engine.can_check_or_call():
            game_engine.check_or_call()
        elif game_engine.can_fold():
            game_engine.fold()

    # Check if hand is over
    if not game_engine.game_state.status:
        # Get final hand information
        hole_cards = game_engine.get_hole_cards(player_index)
        board_cards = game_engine.get_board_cards()
        payout = game_engine.get_player_stack(player_index) - 100

        # Calculate final EV and guidance
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            hole_cards,
            board_cards,
            0,
            0,  # No active bet at showdown
        )

        # Get actual hand information
        hole_cards_str = (
            ", ".join(str(c) for c in hole_cards) if hole_cards else "Unknown"
        )
        board_cards_str = (
            ", ".join(str(c) for c in board_cards) if board_cards else "No board"
        )

        # Determine actual hand type
        hand_type = "Hand Complete"
        if hole_cards and len(hole_cards) == 2:
            c1, c2 = hole_cards
            if hasattr(c1, "rank") and hasattr(c2, "rank"):
                if str(c1)[0] == str(c2)[0]:  # Pair
                    hand_type = f"Pocket {str(c1)[0]}s"
                elif str(c1)[1] == str(c2)[1]:  # Suited
                    hand_type = f"{str(c1)[0]}{str(c2)[0]} suited"
                else:  # Offsuit
                    hand_type = f"{str(c1)[0]}{str(c2)[0]} offsuit"

        # Add final hand to history and database
        hand_data = {
            "vector": vectorize_state(hole_cards, board_cards, 0, 0),
            "ev_penalty": ev_penalty,
            "payout": payout,
            "hand_type": hand_type,
            "cluster": "Hand Complete",
            "hole_cards": hole_cards_str,
            "board_cards": board_cards_str,
            "action": "Hand End",
            "perfection_score": perfection_score,
            "street": "Final",
        }

        # Add to session state for immediate use
        st.session_state.hands.append(hand_data)

        # Save to database
        st.session_state.db.save_hand(hand_data, st.session_state.session_id)

        # Increment counter for UMAP refitting
        st.session_state.hands_since_last_umap_fit += 1

        # Check if we need to refit UMAP (every 20 hands)
        if st.session_state.hands_since_last_umap_fit >= 20:
            st.session_state.umap_needs_refit = True

        # Show hand over screen
        render_hand_over_screen(payout, ev_penalty, perfection_score, guide)

        # Reset game for new hand (optional - user can click New Hand)
        # Commented out to allow users to see final state
        # st.session_state.game_engine.reset_game()

    # 3D Visualization Section
    st.subheader("📊 Strategy Matrix")

    # Prepare data for plotting
    hands_df, vectors = prepare_dataframe_for_plotting(st.session_state.hands)

    # Apply UMAP transformation with performance optimization
    if len(vectors) > 0:
        # Check if we need to refit the UMAP model (every 20 hands)
        if st.session_state.umap_needs_refit or not st.session_state.umap_is_fitted:
            # Re-fit the UMAP model with all available data
            umap_results = apply_umap_transformation(
                vectors, st.session_state.umap_reducer, is_fitted=False
            )
            st.session_state.umap_is_fitted = True
            st.session_state.umap_needs_refit = False
            st.session_state.hands_since_last_umap_fit = 0  # Reset counter
        else:
            # Just transform using the existing fitted model (faster)
            umap_results = apply_umap_transformation(
                vectors, st.session_state.umap_reducer, is_fitted=True
            )
    else:
        umap_results = np.array([]).reshape(0, 3)

    # Create and display plot
    fig = create_3d_scatter_plot(hands_df, umap_results)
    st.plotly_chart(fig, width="stretch")

    # Display hand history if we have real data
    if not hands_df.empty and "hole_cards" in hands_df.columns:
        display_hand_history(hands_df)

    # Control buttons
    col1, col2, col3 = st.columns([1, 1, 4])

    with col1:
        if st.button("New Hand"):
            st.session_state.game_engine = PokerGame()
            # Clear hands for fresh start (optional)
            # st.session_state.hands = []
            st.rerun()

    with col2:
        if st.button("Reset Analytics"):
            st.session_state.hands = []
            st.session_state.umap_is_fitted = False
            st.session_state.umap_reducer = create_umap_reducer()
            st.rerun()


if __name__ == "__main__":
    main()
