# 🃏 Poker Vector Space Visualizer (PVSV)

> **Core Philosophy: Analysis over Aesthetics**  
> This is a Strategic Laboratory, not a commercial game. The UI is built for data density and decision-auditing. The objective is to visualize the geometry of your strategy, not to provide a high-fidelity gaming experience.

![Poker Vector Space Visualizer Screenshot](poker_matrix.html)

## 🎯 Ideal User

This tool is designed for **Poker Researchers, Data Scientists, and Serious Players** seeking to identify "topological leaks" in their decision-making through rigorous, data-driven analysis of their poker strategy in vector space.

## 🚀 Key Features

- **Vectorized State Analysis**: Each poker game state is converted to a 114-dimensional vector for deep strategic analysis.
- **Persistent SQLite Storage**: All hands, actions, and results are stored locally, allowing you to resume sessions and analyze long-term trends.
- **Real-time 3D UMAP Strategy Matrix**: Visualize your poker strategy in a neon cyberpunk 3D vector space, where similar strategies cluster together.
- **Perfect Hand Guide**: Get real-time feedback on your decisions with EV penalty calculations and optimal play suggestions.
- **Modular Architecture**: Clean separation of concerns for maintainability and extensibility.
- **Comprehensive Test Suite**: Verified core logic with pytest.

## 📦 Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/yourusername/poker-vector-space-visualizer.git
   cd poker-vector-space-visualizer
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## ▶️ Usage

Run the application:
```bash
streamlit run poker_matrix_modular.py
```

### How to Play

1. You are Player 0 (human) playing against a simple AI opponent (Player 1).
2. Use the buttons to make your move: Check/Call, Bet/Raise, or Fold.
3. After each action, see the **Perfect Hand Guide** for feedback on your decision.
4. Click **New Hand** to start a fresh hand.
5. Click **Reset Analytics** to clear your history and restart the 3D analysis.

### 📊 Interpreting the 3D Strategy Matrix

The neon 3D plot visualizes your poker strategy using UMAP dimensionality reduction on your 114-dimensional state vectors.

- **Axes (X, Y, Z)**: 
  - These are the three UMAP components that capture the most variance in your poker state vectors.
  - Similar strategies (similar vectors) cluster together in vector space.

- **Color**: 
  - **Red** = Negative EV penalty (bad decisions, you lost value compared to optimal play)
  - **Green** = Positive EV penalty (good decisions, you gained value compared to optimal play)
  - The color bar shows the exact EV penalty in big blinds (bb).

- **Size**: 
  - Represents the magnitude of your payout (win/loss) for that hand.
  - Larger dots = bigger wins/losses (absolute value of chips won/lost).

- **Symbols**: 
  - Different symbols represent different action clusters (e.g., check/call, bet/raise, fold).

- **Hover Over Dots**: 
  - See detailed information about each hand: hole cards, board cards, action taken, hand type, perfection score, and more.

### 💡 Tips for Improvement

1. Aim for high perfection scores (90%+).
2. Notice patterns in the 3D vector space - where do your good/bad decisions cluster?
3. Use the perfect hand guide to learn optimal strategies.
4. Review your hand history to identify leaks in your game.
5. Use the "Reset Analytics" button to start fresh when experimenting with new strategies.

## 🔧 Technical Constraints & Intent

- **AI Opponent**: The AI is a deterministic baseline for expected value (EV) calculation, designed to provide consistent optimal play comparisons rather than simulate human-like adversarial behavior.
- **Streamlit Interface**: Optimized for analytical dashboarding, which involves script-wide reruns on every action to ensure data consistency and real-time updates to the 3D visualization and hand history.

## 🏗️ Architecture

The project follows a modular structure:

```
src/
├── __init__.py
├── game_engine.py      # Poker game logic and state management
├── vector_analysis.py  # State vectorization and EV calculations
├── visualization.py    # 3D plotting and UI components
└── database.py         # Persistent storage with SQLite
```

- **poker_matrix_modular.py**: Main Streamlit application that ties all modules together.
- **Requirements**: See `requirements.txt` for exact versions.

## 🧪 Testing

Run the test suite to verify core logic:
```bash
python -m pytest tests/ -v
```

The suite includes tests for:
- Game engine logic (state management, actions)
- Vector analysis (state encoding, EV estimation, hand strength)
- Edge cases and error handling

## 🗺️ Future Roadmap: Bulk Hand Ingestion

The SQLite schema is engineered for scalability, capable of efficiently storing and querying millions of poker hands. The "Gold State" for PVSV is achieved through importing real hand history (e.g., from hand history files or databases) to enable deep historical audits and longitudinal strategy analysis.

## 📜 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎨 Credits

Inspired by the cyberpunk aesthetic and the desire to visualize poker strategy in a meaningful way.

Built with:
- [Streamlit](https://streamlit.io)
- [Pokerkit](https://pokerkit.readthedocs.io)
- [Treys](https://github.com/treys/treys)
- [UMAP](https://github.com/lmcinnes/umap)
- [Plotly](https://plotly.com/python)
- [NumPy](https://numpy.org)
- [Pandas](https://pandas.pydata.org)

---

**May your vectors be optimal and your EV always positive!** 🀄