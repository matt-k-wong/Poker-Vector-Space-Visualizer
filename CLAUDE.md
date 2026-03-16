# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

**Run the Game:**
```bash
streamlit run poker_matrix.py
```

**Install Dependencies (if needed):**
```bash
pip install streamlit pokerkit treys numpy pandas umap-learn plotly
```

**Python Environment:**
- Python 3.11.9+ required
- Uses pyenv for Python version management

## Codebase Architecture

### Core Components

**poker_matrix.py** - Main Streamlit application with three key systems:

1. **Game Engine** (`poker_matrix.py:11-40`)
   - Uses `pokerkit.NoLimitTexasHoldem` for game state management
   - Manages session state including game history and UMAP clustering
   - Handles 2-player No-Limit Texas Hold'em

2. **Strategy Analysis** (`poker_matrix.py:42-76`)
   - `vectorize_state()` - Converts game states to 114-dimensional vectors
   - `estimate_ev_and_guide()` - Calculates EV penalties and provides optimal play guidance
   - Uses `treys.Evaluator` for hand strength evaluation

3. **3D Visualization** (`poker_matrix.py:145-207`)
   - UMAP dimensionality reduction for 3D plotting
   - Plotly 3D scatter plots with cyberpunk styling
   - Real-time clustering of poker decisions

### Data Flow

```
Game State → Vector Encoding → EV Analysis → 3D Visualization
     ↓              ↓              ↓              ↓
 Session State → Hand History → Perfect Guide → Matrix Plot
```

### Key Libraries

- **pokerkit**: Texas Hold'em game engine and card handling
- **treys**: Fast poker hand evaluation
- **umap-learn**: Dimensionality reduction for visualization
- **streamlit**: Web interface and session management
- **plotly**: Interactive 3D plotting

## Project Structure

- `poker_matrix.py` - Main application (217 lines)
- `README.md` - User documentation with quick start
- `poker_analytics_system.md` - Detailed system design document for future vector database implementation
- `mkw_archive/create_vector_poker.sh` - Setup script for dependencies

## Vector System Design

The codebase implements a real-time vector-based poker analysis system:

- **Vector Encoding**: Each game state becomes a 114-dimensional vector (52 hole cards + 52 board cards + betting/position data)
- **EV Tracking**: Measures Expected Value penalty for each decision vs. optimal play
- **3D Clustering**: UMAP reduces vectors to 3D space for visual pattern recognition
- **Decision Analysis**: Real-time feedback on play quality with "perfection scores"

## Session Management

The application maintains persistent state across user interactions:
- Game history stored in `st.session_state.hands`
- UMAP clustering model persists across hands
- 3D visualization updates incrementally with new data points

## Future Development Notes

The `poker_analytics_system.md` document outlines a comprehensive vector database expansion plan including:
- SQLite schema for hand tracking and analytics
- Advanced ML models for decision prediction
- CLI interface for batch analysis
- Professional analytics dashboard

This represents a foundation for a larger poker analytics system with vector database capabilities.