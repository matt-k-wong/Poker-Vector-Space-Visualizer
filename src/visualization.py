"""
Visualization module for 3D plotting and UI components.
"""

import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from typing import List, Dict, Any, Optional, Tuple
import streamlit as st


def create_umap_reducer(
    n_components: int = 3,
    random_state: int = 42,
    n_neighbors: int = 15,
    min_dist: float = 0.2,
) -> UMAP:
    """
    Create a UMAP reducer for dimensionality reduction.

    Args:
        n_components: Number of dimensions to reduce to
        random_state: Random seed for reproducibility
        n_neighbors: Number of neighbors to consider
        min_dist: Minimum distance between points

    Returns:
        Configured UMAP reducer
    """
    return UMAP(
        n_components=n_components,
        random_state=random_state,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
    )


def prepare_dataframe_for_plotting(
    hands_data: List[Dict[str, Any]],
) -> Tuple[pd.DataFrame, np.ndarray]:
    """
    Convert hands data to DataFrame suitable for plotting.

    Args:
        hands_data: List of dictionaries containing hand information

    Returns:
        Tuple of (DataFrame with hand data, numpy array of vectors)
    """
    if not hands_data:
        # Return empty DataFrame with expected columns
        empty_df = pd.DataFrame(
            {
                "x": [],
                "y": [],
                "z": [],
                "ev_penalty": [],
                "payout": [],
                "hand_type": [],
                "cluster": [],
                "size_val": [],
            }
        )
        return empty_df, np.array([]).reshape(0, 114)

    df = pd.DataFrame(hands_data)

    # Extract vectors for UMAP
    if "vector" in df.columns and len(df) > 0:
        try:
            vectors = np.vstack(df["vector"].values)
            return df, vectors
        except Exception:
            # If vector extraction fails, return empty arrays
            return df, np.array([]).reshape(0, 114)
    else:
        return df, np.array([]).reshape(0, 114)


def apply_umap_transformation(
    vectors: np.ndarray, umap_reducer: UMAP, is_fitted: bool = False
) -> np.ndarray:
    """
    Apply UMAP transformation to vectors.

    Args:
        vectors: Input vectors to transform
        umap_reducer: UMAP reducer instance
        is_fitted: Whether the reducer is already fitted

    Returns:
        Transformed vectors in reduced space
    """
    if len(vectors) == 0:
        return np.array([]).reshape(0, 3)

    try:
        if is_fitted:
            return umap_reducer.transform(vectors)
        else:
            return umap_reducer.fit_transform(vectors)
    except Exception:
        # Return zeros if transformation fails
        return np.zeros((len(vectors), 3))


def create_3d_scatter_plot(
    df: pd.DataFrame,
    umap_results: np.ndarray,
    title: str = "Live Poker Matrix: 3D Strategy Clusters",
) -> Any:
    """
    Create a 3D scatter plot for visualization.

    Args:
        df: DataFrame containing hand data
        umap_results: UMAP transformed coordinates
        title: Plot title

    Returns:
        Plotly figure object
    """
    # Handle empty data case
    if len(df) == 0 or len(umap_results) == 0:
        # Create empty plot with proper structure
        df_plot = pd.DataFrame(
            {
                "x": [0],
                "y": [0],
                "z": [0],
                "ev_penalty": [0],
                "payout": [5],
                "hand_type": ["No Data"],
                "cluster": ["No Data"],
                "size_val": [5],
            }
        )
        umap_results = np.array([[0, 0, 0]])
    else:
        df_plot = df.copy()
        df_plot["x"] = umap_results[:, 0]
        df_plot["y"] = umap_results[:, 1]
        df_plot["z"] = umap_results[:, 2]

        # Ensure positive sizes for plotting
        df_plot["size_val"] = np.abs(df_plot["payout"]) + 5

    # Define hover data columns
    base_hover_cols = ["hand_type", "ev_penalty", "payout", "cluster"]
    extra_hover_cols = [
        "hole_cards",
        "board_cards",
        "action",
        "perfection_score",
        "street",
    ]
    hover_data_cols = base_hover_cols.copy()

    # Add extra columns if they exist
    for col in extra_hover_cols:
        if col in df_plot.columns:
            hover_data_cols.append(col)

    # Define neon color scale
    neon_colors = [[0, "#FF073A"], [0.5, "#00FFFF"], [1, "#39FF14"]]

    # Create the plot
    fig = px.scatter_3d(
        df_plot,
        x="x",
        y="y",
        z="z",
        color="ev_penalty",
        size="size_val",
        symbol="cluster",
        hover_data=hover_data_cols,
        color_continuous_scale=neon_colors,
        title="Live Poker Vector Space: 3D Strategy Clusters",
        opacity=0.9,
        size_max=25,
    )

    # Update layout for dark theme and styling
    # Prepare buttons for updatemenus
    buttons_list = [
        dict(
            label="All Vibes",
            method="update",
            args=[{"visible": [True] * len(df_plot)}],
        )
    ]

    if len(df_plot) > 0:
        for c in df_plot["cluster"].unique():
            buttons_list.append(
                dict(
                    label=f"Show {c}",
                    method="update",
                    args=[{"visible": df_plot["cluster"] == c}],
                )
            )

    fig.update_layout(
        template="plotly_dark",
        width=1000,
        height=800,
        title_font=dict(size=24, family="Arial Black", color="#39FF14"),
        scene=dict(
            xaxis_title="X",
            yaxis_title="Y",
            zaxis_title="Z",
            xaxis=dict(backgroundcolor="#1A1A1A", gridcolor="#00FFFF"),
            yaxis=dict(backgroundcolor="#1A1A1A", gridcolor="#00FFFF"),
            zaxis=dict(backgroundcolor="#1A1A1A", gridcolor="#00FFFF"),
            aspectmode="cube",
        ),
        coloraxis_colorbar=dict(
            title="EV Penalty (bb)", tickfont=dict(color="#00FFFF")
        ),
        showlegend=True,
        margin=dict(l=0, r=0, t=50, b=0),
        updatemenus=[
            dict(
                buttons=buttons_list,
                direction="down",
                showactive=True,
                x=0.1,
                xanchor="left",
                y=1.1,
                yanchor="top",
                bgcolor="#1A1A1A",
                font=dict(color="#39FF14"),
            )
        ],
    )

    # Add annotation
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.95,
        text="🎯 Hover over dots to see your hands & cards! 🃏 Colors show EV penalty (red=bad, green=good)",
        showarrow=False,
        font=dict(size=14, color="#39FF14", family="Arial Black"),
    )

    return fig


def display_hand_history(df: pd.DataFrame, num_recent: int = 10) -> None:
    """
    Display recent hand history in a table format.

    Args:
        df: DataFrame containing hand data
        num_recent: Number of recent hands to display
    """
    if df.empty or "hole_cards" not in df.columns:
        st.info("No hand history to display yet.")
        return

    # Get recent hands
    recent_hands = df.tail(num_recent).copy()

    if recent_hands.empty:
        st.info("No hand history to display yet.")
        return

    # Create display table
    display_columns = [
        "hole_cards",
        "board_cards",
        "action",
        "hand_type",
        "perfection_score",
        "ev_penalty",
        "payout",
    ]

    # Filter to only columns that exist
    available_columns = [col for col in display_columns if col in recent_hands.columns]
    display_df = recent_hands[available_columns].copy()

    # Rename columns for display
    column_mapping = {
        "hole_cards": "Hole Cards",
        "board_cards": "Board",
        "action": "Action",
        "hand_type": "Hand Type",
        "perfection_score": "Perfection %",
        "ev_penalty": "EV Penalty",
        "payout": "Payout",
    }

    display_df = display_df.rename(columns=column_mapping)

    # Round numeric columns
    if "Perfection %" in display_df.columns:
        display_df["Perfection %"] = display_df["Perfection %"].round(1)
    if "EV Penalty" in display_df.columns:
        display_df["EV Penalty"] = display_df["EV Penalty"].round(2)
    if "Payout" in display_df.columns:
        display_df["Payout"] = display_df["Payout"].round(0)

    st.dataframe(display_df, use_container_width=True)


def render_game_state_info(game_engine: Any, player_index: int = 0) -> None:
    """
    Render current game state information in Streamlit.

    Args:
        game_engine: PokerGame instance
        player_index: Index of the player to show info for
    """
    st.write(
        f"**Your Hand**: {', '.join(str(c) for c in game_engine.get_hole_cards(player_index)) if game_engine.get_hole_cards(player_index) else 'Waiting'}"
    )

    board_cards = game_engine.get_board_cards()
    if board_cards:
        st.write(f"**Board**: {', '.join(str(c) for c in board_cards)}")

    st.write(
        f"**Pot**: ${game_engine.get_pot_amount()} | **Your Stack**: ${game_engine.get_player_stack(player_index)}"
    )


def render_action_buttons(
    game_engine: Any, player_index: int = 0
) -> Tuple[Optional[str], Optional[float]]:
    """
    Render action buttons and return user action.

    Args:
        game_engine: PokerGame instance
        player_index: Index of the human player

    Returns:
        Tuple of (action_taken, bet_amount)
    """
    action_taken = None
    bet_amount = None

    if (
        not game_engine.game_state.status
        or game_engine.get_current_player() != player_index
    ):
        return action_taken, bet_amount

    st.subheader("Your Move:")
    cols = st.columns(4)

    # Check/Call button
    if game_engine.can_check_or_call() and cols[0].button(
        "Check/Call", key="check_call"
    ):
        game_engine.check_or_call()
        action_taken = "check_call"

    # Bet/Raise controls
    if game_engine.can_bet_or_raise():
        min_bet = game_engine.get_min_bet()
        max_bet = game_engine.get_max_bet()
        bet_amount = cols[1].number_input(
            "Bet Amount",
            min_value=min_bet,
            max_value=max_bet,
            value=min_bet,
            key="bet_input",
        )
        if cols[1].button("Bet/Raise", key="bet"):
            game_engine.bet_or_raise(bet_amount)
            action_taken = "bet"

    # Fold button
    if game_engine.can_fold() and cols[2].button("Fold", key="fold"):
        game_engine.fold()
        action_taken = "fold"

    return action_taken, bet_amount


def render_perfect_guide(
    ev_penalty: float, perfection_score: float, guide: str
) -> None:
    """
    Render the perfect hand guide section.

    Args:
        ev_penalty: EV penalty in big blinds
        perfection_score: Perfection score (0-100)
        guide: Guidance text to display
    """
    st.subheader("Perfect Hand Guide")
    st.metric("Perfection Score", f"{perfection_score:.1f}%")
    st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
    st.write(f"**Tip**: {guide}")


def render_hand_over_screen(
    payout: float, ev_penalty: float, perfection_score: float, guide: str
) -> None:
    """
    Render hand over screen.

    Args:
        payout: Net win/loss for the hand
        ev_penalty: EV penalty in big blinds
        perfection_score: Perfection score (0-100)
        guide: Guidance text to display
    """
    st.write("**Hand Over!**")
    st.subheader("Perfect Hand Guide")
    st.metric("Perfection Score", f"{perfection_score:.1f}%")
    st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
    st.write(f"**Tip**: {guide}")
