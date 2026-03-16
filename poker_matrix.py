import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from pokerkit import NoLimitTexasHoldem, Deck, Card
from treys import Evaluator, Card as TreysCard
import random

# NOTE: This is the original monolithic version.
# For improved modularity and maintainability, see poker_matrix_modular.py
# which separates concerns into game_engine.py, vector_analysis.py, and visualization.py

# Initialize session state
if "game" not in st.session_state:
    from pokerkit import Automation

    st.session_state.game = NoLimitTexasHoldem.create_state(
        (
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
        ),
        False,
        [],
        [1, 2],
        100,
        [100, 100],
        2,
    )
    st.session_state.hands = []
    # Precomputed sample hands for demo (will be replaced with real hands)
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

    st.session_state.past_vectors = np.random.rand(100, 114)
    st.session_state.past_ev_penalties = np.random.uniform(-1, 1, 100)
    st.session_state.past_payouts = np.random.uniform(0, 100, 100)
    st.session_state.past_hand_types = random.choices(
        ["Pair", "Suited", "Bluff", "Premium"], k=100
    )
    st.session_state.umap = UMAP(
        n_components=3, random_state=42, n_neighbors=15, min_dist=0.2
    )


# Vectorize game state
def vectorize_state(game, player=0):
    vector = np.zeros(114)
    if game.hole_cards and game.hole_cards[player]:
        for card in game.hole_cards[player]:
            vector[Card.to_id(card)] = 1
    if game.board_cards:
        for card in game.board_cards[0]:
            vector[52 + Card.to_id(card)] = 1
    vector[104] = (
        game.bets[player] / game.total_pot_amount if game.total_pot_amount else 0
    )
    vector[105 + player] = 1
    return vector


# Estimate EV and perfection guide
def estimate_ev_and_guide(game, player=0, action=None, bet_amount=None):
    if not game.hole_cards or not game.hole_cards[player]:
        return 0, 100, "No hand yet."
    evaluator = Evaluator()
    hand = [TreysCard(str(card)) for card in game.hole_cards[player]]
    board = (
        [TreysCard(str(card)) for card in game.board_cards[0]]
        if game.board_cards
        else []
    )
    rank = evaluator.evaluate(board, hand) if board else 1000
    equity = 1 - (rank / 7462)

    pot_total = game.total_pot_amount
    pot_odds = game.bets[1] / pot_total if game.bets[1] and pot_total else 0
    optimal_action = "fold"
    optimal_bet = 0
    if equity > 0.7:
        optimal_action = "bet"
        optimal_bet = pot_total * 0.5
    elif equity > pot_odds:
        optimal_action = "call"
    optimal_ev = equity * pot_total if optimal_action != "fold" else 0

    actual_ev = 0
    if action == "bet" and bet_amount:
        actual_ev = equity * (pot_total + bet_amount) - bet_amount
    elif action == "call":
        actual_ev = equity * pot_total - game.bets[1]
    elif action == "fold":
        actual_ev = 0

    ev_penalty = actual_ev - optimal_ev
    perfection_score = max(0, 100 - abs(ev_penalty) * 50)
    guide = (
        f"Optimal: {optimal_action} {optimal_bet if optimal_action == 'bet' else ''}. "
    )
    if action and perfection_score < 90:
        guide += f"Your {action} loses {ev_penalty:.2f} bb vs. optimal."
    else:
        guide += "Nailed it!"
    return ev_penalty, perfection_score, guide


# Streamlit UI
st.title("Poker Matrix: Cyberpunk Poker Trainer")
st.markdown(
    "**Play No-Limit Hold'em, optimize with a perfect hand guide, and explore your strategy in a neon 3D matrix!**"
)

# Game controls
game = st.session_state.game
player = game.actor_index if game.actor_index is not None else 0
st.write(
    f"**Your Hand**: {', '.join(str(c) for c in game.hole_cards[player]) if game.hole_cards and game.hole_cards[player] else 'Waiting'}"
)
if game.board_cards:
    st.write(f"**Board**: {', '.join(str(c) for c in game.board_cards[0])}")
st.write(f"**Pot**: ${game.total_pot_amount} | **Your Stack**: ${game.stacks[player]}")

# Action buttons - only show for human player (player 0)
if game.status and game.actor_index == 0:
    st.subheader("Your Move:")
    cols = st.columns(4)
    action_taken = None
    bet_amount = None

    if game.can_check_or_call and cols[0].button("Check/Call", key="check_call"):
        game.check_or_call()
        action_taken = "check_call"

    if game.can_complete_bet_or_raise_to:
        min_bet = game.min_completion_betting_or_raising_to_amount
        max_bet = game.max_completion_betting_or_raising_to_amount
        bet_amount = cols[1].number_input(
            "Bet Amount",
            min_value=min_bet,
            max_value=max_bet,
            value=min_bet,
            key="bet_input",
        )
        if cols[1].button("Bet/Raise", key="bet"):
            game.complete_bet_or_raise_to(bet_amount)
            action_taken = "bet"

    if game.can_fold and cols[2].button("Fold", key="fold"):
        game.fold()
        action_taken = "fold"

    if action_taken:
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(
            game, player, action_taken, bet_amount
        )

        # Get actual hand information
        hole_cards_str = (
            ", ".join(str(c) for c in game.hole_cards[player])
            if game.hole_cards and game.hole_cards[player]
            else "Unknown"
        )
        board_cards_str = (
            ", ".join(str(c) for c in game.board_cards[0])
            if game.board_cards
            else "Preflop"
        )

        # Determine actual hand type
        hand_type = "Unknown"
        if (
            game.hole_cards
            and game.hole_cards[player]
            and len(game.hole_cards[player]) == 2
        ):
            c1, c2 = game.hole_cards[player]
            if str(c1)[0] == str(c2)[0]:  # Pair
                hand_type = f"Pocket {str(c1)[0]}s"
            elif str(c1)[1] == str(c2)[1]:  # Suited
                hand_type = f"{str(c1)[0]}{str(c2)[0]} suited"
            else:  # Offsuit
                hand_type = f"{str(c1)[0]}{str(c2)[0]} offsuit"

        st.session_state.hands.append(
            {
                "vector": vectorize_state(game),
                "ev_penalty": ev_penalty,
                "payout": game.stacks[player] - 100,
                "hand_type": hand_type,
                "cluster": "Your Hand",
                "hole_cards": hole_cards_str,
                "board_cards": board_cards_str,
                "action": action_taken,
                "perfection_score": perfection_score,
                "street": "Preflop"
                if not game.board_cards
                else f"{len(game.board_cards[0])} cards",
            }
        )
        st.subheader("Perfect Hand Guide")
        st.metric("Perfection Score", f"{perfection_score:.1f}%")
        st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
        st.write(f"**Tip**: {guide}")

# AI moves - only act if it's the AI's turn (player 1)
if game.status and game.actor_index == 1:
    # Simple AI logic - just check/call or fold
    if game.can_check_or_call:
        game.check_or_call()
    elif game.can_fold:
        game.fold()

# Check if hand is over
if not game.status:
    st.write("**Hand Over!**")
    payout = game.stacks[player] - 100
    ev_penalty, perfection_score, guide = estimate_ev_and_guide(game)

    # Get actual hand information for hand over
    hole_cards_str = (
        ", ".join(str(c) for c in game.hole_cards[player])
        if game.hole_cards and game.hole_cards[player]
        else "Unknown"
    )
    board_cards_str = (
        ", ".join(str(c) for c in game.board_cards[0])
        if game.board_cards
        else "No board"
    )

    # Determine actual hand type
    hand_type = "Hand Complete"
    if (
        game.hole_cards
        and game.hole_cards[player]
        and len(game.hole_cards[player]) == 2
    ):
        c1, c2 = game.hole_cards[player]
        if str(c1)[0] == str(c2)[0]:  # Pair
            hand_type = f"Pocket {str(c1)[0]}s"
        elif str(c1)[1] == str(c2)[1]:  # Suited
            hand_type = f"{str(c1)[0]}{str(c2)[0]} suited"
        else:  # Offsuit
            hand_type = f"{str(c1)[0]}{str(c2)[0]} offsuit"

    st.session_state.hands.append(
        {
            "vector": vectorize_state(game),
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
    )
    st.subheader("Perfect Hand Guide")
    st.metric("Perfection Score", f"{perfection_score:.1f}%")
    st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
    st.write(f"**Tip**: {guide}")

# 3D Plot
df = pd.DataFrame(st.session_state.hands)
if not df.empty:
    vectors = np.vstack(df["vector"])
    umap_results = st.session_state.umap.fit_transform(vectors)
    df["x"] = umap_results[:, 0]
    df["y"] = umap_results[:, 1]
    df["z"] = umap_results[:, 2]
else:
    df = pd.DataFrame(
        {
            "x": st.session_state.past_vectors[:, 0],
            "y": st.session_state.past_vectors[:, 1],
            "z": st.session_state.past_vectors[:, 2],
            "ev_penalty": st.session_state.past_ev_penalties,
            "payout": np.abs(st.session_state.past_payouts),  # Ensure positive values
            "hand_type": st.session_state.past_hand_types,
            "cluster": st.session_state.past_hand_types,
        }
    )

# Update hover data if new columns exist
hover_data_cols = ["hand_type", "ev_penalty", "payout", "cluster"]
if "hole_cards" in df.columns:
    hover_data_cols.extend(
        ["hole_cards", "board_cards", "action", "perfection_score", "street"]
    )

# Ensure all sizes are positive for plotting
df = df.copy()  # Make a copy to avoid warnings
df["size_val"] = np.abs(df["payout"]) + 5  # Add 5 to make all dots visible

neon_colors = [[0, "#FF073A"], [0.5, "#00FFFF"], [1, "#39FF14"]]
fig = px.scatter_3d(
    df,
    x="x",
    y="y",
    z="z",
    color="ev_penalty",
    size="size_val",
    symbol="cluster",
    hover_data=hover_data_cols,
    color_continuous_scale=neon_colors,
    title="Live Poker Matrix: 3D Strategy Clusters",
    opacity=0.9,
    size_max=25,
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
    coloraxis_colorbar=dict(title="EV Penalty (bb)", tickfont=dict(color="#00FFFF")),
    showlegend=True,
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[
        dict(
            buttons=[
                dict(
                    label="All Vibes",
                    method="update",
                    args=[{"visible": [True] * len(df)}],
                ),
                *[
                    dict(
                        label=f"Show {c}",
                        method="update",
                        args=[{"visible": df["cluster"] == c}],
                    )
                    for c in df["cluster"].unique()
                ],
            ],
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

fig.add_annotation(
    xref="paper",
    yref="paper",
    x=0.5,
    y=0.95,
    text="🎯 Hover over dots to see your hands & cards! 🃏 Colors show EV penalty (red=bad, green=good)",
    showarrow=False,
    font=dict(size=14, color="#39FF14", family="Arial Black"),
)

st.plotly_chart(fig)

# Show hands table if we have real data
if not df.empty and "hole_cards" in df.columns:
    st.subheader("🃏 Your Hand History")
    # Show only the last 10 hands for readability
    recent_hands = df.tail(10).copy()
    if not recent_hands.empty:
        # Create a clean display table
        display_df = recent_hands[
            [
                "hole_cards",
                "board_cards",
                "action",
                "hand_type",
                "perfection_score",
                "ev_penalty",
                "payout",
            ]
        ].copy()
        display_df.columns = [
            "Hole Cards",
            "Board",
            "Action",
            "Hand Type",
            "Perfection %",
            "EV Penalty",
            "Payout",
        ]
        display_df = display_df.round({"Perfection %": 1, "EV Penalty": 2, "Payout": 0})
        st.dataframe(display_df, use_container_width=True)

# New hand button
if st.button("New Hand"):
    from pokerkit import Automation

    st.session_state.game = NoLimitTexasHoldem.create_state(
        (
            Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.HOLE_DEALING,
            Automation.BOARD_DEALING,
        ),
        False,
        [],
        [1, 2],
        100,
        [100, 100],
        2,
    )
    st.rerun()

# Save HTML
fig.write_html(f"{st.session_state.get('project_dir', '.')}/poker_matrix.html")
