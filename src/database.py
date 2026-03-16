"""
Database module for persistent storage of poker hands and analytics.
Uses SQLite for simplicity and zero-configuration deployment.
"""

import sqlite3
import json
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import os


class PokerDatabase:
    """Handles all database operations for the poker matrix application."""

    def __init__(self, db_path: str = "data/poker_matrix.db"):
        """
        Initialize the database connection and create tables if needed.

        Args:
            db_path: Path to the SQLite database file
        """
        self.db_path = db_path
        # Ensure data directory exists
        os.makedirs(os.path.dirname(db_path), exist_ok=True)
        self.init_database()

    def init_database(self) -> None:
        """Create database tables if they don't exist."""
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()

            # Hands table - stores each poker hand played
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS hands (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                    session_id TEXT,
                    vector BLOB,  -- 114-dimensional vector as binary
                    ev_penalty REAL,
                    payout REAL,
                    hand_type TEXT,
                    cluster TEXT,
                    hole_cards TEXT,
                    board_cards TEXT,
                    action TEXT,
                    perfection_score REAL,
                    street TEXT,
                    game_state_hash TEXT  -- Hash of game state for deduplication
                )
            """)

            # Game statistics table - aggregates for quick dashboard views
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS game_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    date DATE DEFAULT CURRENT_DATE,
                    total_hands INTEGER DEFAULT 0,
                    avg_perfection_score REAL DEFAULT 0.0,
                    total_payout REAL DEFAULT 0.0,
                    avg_ev_penalty REAL DEFAULT 0.0,
                    best_hand_type TEXT,
                    worst_hand_type TEXT
                )
            """)

            # Create indexes for better query performance
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hands_timestamp ON hands(timestamp)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hands_session ON hands(session_id)
            """)
            cursor.execute("""
                CREATE INDEX IF NOT EXISTS idx_hands_ev_penalty ON hands(ev_penalty)
            """)

            conn.commit()

    def save_hand(self, hand_data: Dict[str, Any], session_id: str = None) -> int:
        """
        Save a hand to the database.

        Args:
            hand_data: Dictionary containing hand information
            session_id: Optional session identifier

        Returns:
            ID of the inserted hand
        """
        if session_id is None:
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Convert vector to binary for storage
        vector_blob = hand_data["vector"].astype(np.float32).tobytes()

        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO hands (
                    session_id, vector, ev_penalty, payout, hand_type, cluster,
                    hole_cards, board_cards, action, perfection_score, street
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    session_id,
                    vector_blob,
                    hand_data["ev_penalty"],
                    hand_data["payout"],
                    hand_data["hand_type"],
                    hand_data["cluster"],
                    hand_data["hole_cards"],
                    hand_data["board_cards"],
                    hand_data["action"],
                    hand_data["perfection_score"],
                    hand_data["street"],
                ),
            )

            hand_id = cursor.lastrowid
            conn.commit()
            return hand_id

    def get_hands(
        self, limit: int = 1000, offset: int = 0, session_id: str = None
    ) -> List[Dict[str, Any]]:
        """
        Retrieve hands from the database.

        Args:
            limit: Maximum number of hands to return
            offset: Number of hands to skip
            session_id: Optional session filter

        Returns:
            List of hand dictionaries
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row  # Enable column access by name
            cursor = conn.cursor()

            if session_id:
                cursor.execute(
                    """
                    SELECT * FROM hands 
                    WHERE session_id = ? 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """,
                    (session_id, limit, offset),
                )
            else:
                cursor.execute(
                    """
                    SELECT * FROM hands 
                    ORDER BY timestamp DESC 
                    LIMIT ? OFFSET ?
                """,
                    (limit, offset),
                )

            rows = cursor.fetchall()
            hands = []

            for row in rows:
                hand = dict(row)
                # Convert vector blob back to numpy array
                vector_blob = hand["vector"]
                hand["vector"] = np.frombuffer(vector_blob, dtype=np.float32)
                hands.append(hand)

            return hands

    def get_hand_count(self, session_id: str = None) -> int:
        """
        Get the total number of hands in the database.

        Args:
            session_id: Optional session filter

        Returns:
            Total count of hands
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.cursor()
            if session_id:
                cursor.execute(
                    "SELECT COUNT(*) FROM hands WHERE session_id = ?", (session_id,)
                )
            else:
                cursor.execute("SELECT COUNT(*) FROM hands")
            return cursor.fetchone()[0]

    def get_recent_hands_for_umap(
        self, limit: int = 100
    ) -> Tuple[np.ndarray, List[Dict]]:
        """
        Get recent hands for UMAP visualization.

        Args:
            limit: Number of recent hands to fetch

        Returns:
            Tuple of (vectors_array, hands_list)
        """
        hands = self.get_hands(limit=limit)
        if not hands:
            return np.array([]).reshape(0, 114), []

        vectors = np.vstack([hand["vector"] for hand in hands])
        return vectors, hands

    def find_similar_hands(
        self, target_vector: np.ndarray, threshold: float = 0.85, limit: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find hands with similar vectors using cosine similarity.

        Args:
            target_vector: The vector to compare against
            threshold: Minimum similarity score (0-1)
            limit: Maximum number of results to return

        Returns:
            List of similar hands with similarity scores
        """
        # For small datasets, we can do this in memory
        # For larger datasets, we'd want to use vector similarity indexes
        hands = self.get_hands(limit=1000)  # Get a reasonable sample

        if not hands:
            return []

        # Calculate cosine similarity
        target_norm = np.linalg.norm(target_vector)
        if target_norm == 0:
            return []

        similarities = []
        for hand in hands:
            hand_vector = hand["vector"]
            hand_norm = np.linalg.norm(hand_vector)
            if hand_norm == 0:
                continue

            similarity = np.dot(target_vector, hand_vector) / (target_norm * hand_norm)
            if similarity >= threshold:
                similarities.append((hand, similarity))

        # Sort by similarity (descending) and return top results
        similarities.sort(key=lambda x: x[1], reverse=True)
        return [hand for hand, _ in similarities[:limit]]

    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """
        Get summary statistics for a session.

        Args:
            session_id: Session identifier

        Returns:
            Dictionary with session statistics
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.cursor()
            cursor.execute(
                """
                SELECT 
                    COUNT(*) as hand_count,
                    AVG(perfection_score) as avg_perfection,
                    AVG(ev_penalty) as avg_ev_penalty,
                    SUM(payout) as total_payout,
                    MIN(timestamp) as first_hand,
                    MAX(timestamp) as last_hand
                FROM hands 
                WHERE session_id = ?
            """,
                (session_id,),
            )

            row = cursor.fetchone()
            if row:
                return dict(row)
            return {}

    def vacuum_database(self) -> None:
        """
        Run SQLite VACUUM command to reclaim unused space and defragment the database file.
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("VACUUM")


# Global database instance
db = PokerDatabase()


def get_database() -> PokerDatabase:
    """Get the global database instance."""
    return db
