#!/bin/bash

# build_vector_poker_game.sh
# Creates a playable poker game with 3D visualization and perfect hand guide

# Set project directory
PROJECT_DIR="$HOME/vector_poker"
mkdir -p "$PROJECT_DIR"

# Check if Homebrew is installed
# if ! command -v brew &> /dev/null; then
#    echo "Installing Homebrew..."
#    /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
# fi

# Install Python 3.9+ and pip
# echo "Installing Python and pip..."
# brew install python3
# python3 -m ensurepip --upgrade
# python3 -m pip install --upgrade pip

# Install dependencies
#echo "Installing Python dependencies..."
# python3 -m pip install streamlit pokerkit treys numpy pandas umap-learn plotly

# Write Python script
cat > "$PROJECT_DIR/poker_matrix.py" << 'EOF'
import streamlit as st
import numpy as np
import pandas as pd
import plotly.express as px
from umap import UMAP
from pokerkit import NoLimitTexasHoldem, Deck, Card
from treys import Evaluator, Card as TreysCard
import random

# Initialize session state
if 'game' not in st.session_state:
    st.session_state.game = NoLimitTexasHoldem.create_state(
        [False] * 2, [None] * 2, [0, 1], [1, 2], 100, [100, 100], 2
    )
    st.session_state.hands = []
    # Precomputed perfect hands (simplified)
    perfect_vectors = np.random.rand(10, 114)
    st.session_state.hands.extend([
        {'vector': v, 'ev_penalty': 0, 'payout': 50, 'hand_type': 'Perfect', 'cluster': 'Perfect'}
        for v in perfect_vectors
    ])
    st.session_state.past_vectors = np.random.rand(100, 114)
    st.session_state.past_ev_penalties = np.random.uniform(-1, 1, 100)
    st.session_state.past_payouts = np.random.uniform(0, 100, 100)
    st.session_state.past_hand_types = random.choices(['Pair', 'Suited', 'Bluff', 'Premium'], k=100)
    st.session_state.umap = UMAP(n_components=3, random_state=42, n_neighbors=15, min_dist=0.2)

# Vectorize game state
def vectorize_state(game, player=0):
    vector = np.zeros(114)
    if game.hole_cards and game.hole_cards[player]:
        for card in game.hole_cards[player]:
            vector[Card.to_id(card)] = 1
    if game.board_cards:
        for card in game.board_cards[0]:
            vector[52 + Card.to_id(card)] = 1
    vector[104] = game.bets[player] / game.pot if game.pot else 0
    vector[105 + game.positions[player]] = 1
    return vector

# Estimate EV and perfection guide
def estimate_ev_and_guide(game, player=0, action=None, bet_amount=None):
    if not game.hole_cards or not game.hole_cards[player]:
        return 0, 100, "No hand yet."
    evaluator = Evaluator()
    hand = [TreysCard(str(card)) for card in game.hole_cards[player]]
    board = [TreysCard(str(card)) for card in game.board_cards[0]] if game.board_cards else []
    rank = evaluator.evaluate(board, hand) if board else 1000
    equity = 1 - (rank / 7462)

    pot_odds = game.bets[1] / game.pot if game.bets[1] and game.pot else 0
    optimal_action = 'fold'
    optimal_bet = 0
    if equity > 0.7:
        optimal_action = 'bet'
        optimal_bet = game.pot * 0.5
    elif equity > pot_odds:
        optimal_action = 'call'
    optimal_ev = equity * game.pot if optimal_action != 'fold' else 0

    actual_ev = 0
    if action == 'bet' and bet_amount:
        actual_ev = equity * (game.pot + bet_amount) - bet_amount
    elif action == 'call':
        actual_ev = equity * game.pot - game.bets[1]
    elif action == 'fold':
        actual_ev = 0

    ev_penalty = actual_ev - optimal_ev
    perfection_score = max(0, 100 - abs(ev_penalty) * 50)
    guide = f"Optimal: {optimal_action} {optimal_bet if optimal_action == 'bet' else ''}. "
    if action and perfection_score < 90:
        guide += f"Your {action} loses {ev_penalty:.2f} bb vs. optimal."
    else:
        guide += "Nailed it!"
    return ev_penalty, perfection_score, guide

# Streamlit UI
st.title("Poker Matrix: Cyberpunk Poker Trainer")
st.markdown("**Play No-Limit Hold'em, optimize with a perfect hand guide, and explore your strategy in a neon 3D matrix!**")

# Game controls
game = st.session_state.game
player = game.actors[0] if game.actors else 0
st.write(f"**Your Hand**: {', '.join(str(c) for c in game.hole_cards[player]) if game.hole_cards and game.hole_cards[player] else 'Waiting'}")
if game.board_cards:
    st.write(f"**Board**: {', '.join(str(c) for c in game.board_cards[0])}")
st.write(f"**Pot**: ${game.pot} | **Your Stack**: ${game.stacks[player]}")

# Action buttons
if game.status and game.is_actor_turn():
    actions = game.get_legal_actions()
    st.subheader("Your Move:")
    cols = st.columns(4)
    action_taken = None
    bet_amount = None
    if 'check' in actions and cols[0].button("Check", key="check"):
        game.check_or_call()
        action_taken = 'check'
    if 'call' in actions and cols[1].button(f"Call ${actions['call'] if isinstance(actions['call'], int) else 0}", key="call"):
        game.check_or_call()
        action_taken = 'call'
    if 'bet' in actions:
        bet_amount = cols[2].number_input("Bet Amount", min_value=actions['bet'][0], max_value=actions['bet'][1], value=actions['bet'][0], key="bet_input")
        if cols[2].button("Bet", key="bet"):
            game.bet(bet_amount)
            action_taken = 'bet'
    if 'fold' in actions and cols[3].button("Fold", key="fold"):
        game.fold()
        action_taken = 'fold'

    if action_taken:
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(game, player, action_taken, bet_amount)
        st.session_state.hands.append({
            'vector': vectorize_state(game),
            'ev_penalty': ev_penalty,
            'payout': game.stacks[player] - 100,
            'hand_type': random.choice(['Pair', 'Suited', 'Bluff', 'Premium']),
            'cluster': 'Your Hand'
        })
        st.subheader("Perfect Hand Guide")
        st.metric("Perfection Score", f"{perfection_score:.1f}%")
        st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
        st.write(f"**Tip**: {guide}")

# AI moves
if game.status and not game.is_actor_turn():
    game.complete_bettings()
    if game.is_terminal():
        st.write("**Hand Over!**")
        payout = game.stacks[player] - 100
        ev_penalty, perfection_score, guide = estimate_ev_and_guide(game)
        st.session_state.hands.append({
            'vector': vectorize_state(game),
            'ev_penalty': ev_penalty,
            'payout': payout,
            'hand_type': random.choice(['Pair', 'Suited', 'Bluff', 'Premium']),
            'cluster': 'Your Hand'
        })
        st.subheader("Perfect Hand Guide")
        st.metric("Perfection Score", f"{perfection_score:.1f}%")
        st.metric("EV Penalty", f"{ev_penalty:.2f} bb")
        st.write(f"**Tip**: {guide}")

# 3D Plot
df = pd.DataFrame(st.session_state.hands)
if not df.empty:
    vectors = np.vstack(df['vector'])
    umap_results = st.session_state.umap.fit_transform(vectors)
    df['x'] = umap_results[:, 0]
    df['y'] = umap_results[:, 1]
    df['z'] = umap_results[:, 2]
else:
    df = pd.DataFrame({
        'x': st.session_state.past_vectors[:, 0],
        'y': st.session_state.past_vectors[:, 1],
        'z': st.session_state.past_vectors[:, 2],
        'ev_penalty': st.session_state.past_ev_penalties,
        'payout': st.session_state.past_payouts,
        'hand_type': st.session_state.past_hand_types,
        'cluster': st.session_state.past_hand_types
    })

neon_colors = [[0, '#FF073A'], [0.5, '#00FFFF'], [1, '#39FF14']]
fig = px.scatter_3d(
    df,
    x='x', y='y', z='z',
    color='ev_penalty', size='payout', symbol='cluster',
    hover_data=['hand_type', 'ev_penalty', 'payout', 'cluster'],
    color_continuous_scale=neon_colors,
    title='Live Poker Matrix: 3D Strategy Clusters',
    opacity=0.9, size_max=25
)

fig.update_layout(
    template='plotly_dark', width=1000, height=800,
    title_font=dict(size=24, family='Arial Black', color='#39FF14'),
    scene=dict(
        xaxis_title='X', yaxis_title='Y', zaxis_title='Z',
        xaxis=dict(backgroundcolor='#1A1A1A', gridcolor='#00FFFF'),
        yaxis=dict(backgroundcolor='#1A1A1A', gridcolor='#00FFFF'),
        zaxis=dict(backgroundcolor='#1A1A1A', gridcolor='#00FFFF'),
        aspectmode='cube'
    ),
    coloraxis_colorbar=dict(title='EV Penalty (bb)', tickfont=dict(color='#00FFFF')),
    showlegend=True,
    margin=dict(l=0, r=0, t=50, b=0),
    updatemenus=[
        dict(
            buttons=[
                dict(label='All Vibes', method='update', args=[{'visible': [True]*len(df)}]),
                *[dict(label=f'Show {c}', method='update', args=[{'visible': df['cluster'] == c}]) for c in df['cluster'].unique()]
            ],
            direction='down', showactive=True,
            x=0.1, xanchor='left', y=1.1, yanchor='top',
            bgcolor='#1A1A1A', font=dict(color='#39FF14')
        )
    ]
)

fig.add_annotation(
    xref='paper', yref='paper', x=0.5, y=0.95,
    text="Spin, Zoom, Hover: Master the Matrix!",
    showarrow=False, font=dict(size=16, color='#39FF14', family='Arial Black')
)

st.plotly_chart(fig)

# New hand button
if st.button("New Hand"):
    st.session_state.game = NoLimitTexasHoldem.create_state(
        [False] * 2, [None] * 2, [0, 1], [1, 2], 100, [100, 100], 2
    )

# Save HTML
fig.write_html(f"{st.session_state.get('project_dir', '.')}/poker_matrix.html")
EOF

# Write README.md
cat > "$PROJECT_DIR/README.md" << 'EOF'
# Vector Poker Matrix

A cyberpunk-styled Texas Hold'em game with a real-time 3D visualization and perfect hand guide.

## Quick Start
1. **Run the Game**:
   ```bash
   cd ~/vector_poker
   streamlit run poker_matrix.py
