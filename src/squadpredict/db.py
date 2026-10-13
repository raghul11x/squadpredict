"""Database layer — SQLite storage for squads, predictions, and real results."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from squadpredict.config import DB_PATH


_SCHEMA = """
CREATE TABLE IF NOT EXISTS squads (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT NOT NULL,
    formation TEXT NOT NULL,
    players_json TEXT NOT NULL,
    source TEXT DEFAULT 'manual',
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    squad_a_id INTEGER REFERENCES squads(id),
    squad_b_id INTEGER REFERENCES squads(id),
    win_prob REAL,
    draw_prob REAL,
    loss_prob REAL,
    predicted_scoreline TEXT,
    model_version TEXT,
    created_at TEXT DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS real_match_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prediction_id INTEGER REFERENCES predictions(id),
    squad_a_id INTEGER REFERENCES squads(id),
    squad_b_id INTEGER REFERENCES squads(id),
    actual_scoreline TEXT NOT NULL,
    actual_outcome TEXT NOT NULL,
    logged_at TEXT DEFAULT (datetime('now'))
);
"""


def _get_connection(db_path: str | Path | None = None) -> sqlite3.Connection:
    """Get a SQLite connection with row factory."""
    path = str(db_path or DB_PATH)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(path)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_db(db_path: str | Path | None = None) -> None:
    """Create database tables if they don't exist."""
    conn = _get_connection(db_path)
    try:
        conn.executescript(_SCHEMA)
        conn.commit()
    finally:
        conn.close()


def save_squad(
    name: str,
    formation: str,
    players_json: str,
    source: str = "manual",
    db_path: str | Path | None = None,
) -> int:
    """Save a squad and return its ID.

    Args:
        name: Squad/team name.
        formation: Formation string (e.g., '4-3-3').
        players_json: JSON string of player list.
        source: 'manual' or 'ocr'.
        db_path: Optional database path override.

    Returns:
        The new squad's ID.
    """
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            "INSERT INTO squads (name, formation, players_json, source) VALUES (?, ?, ?, ?)",
            (name, formation, players_json, source),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def save_prediction(
    squad_a_id: int,
    squad_b_id: int,
    win_prob: float,
    draw_prob: float,
    loss_prob: float,
    predicted_scoreline: str,
    model_version: str,
    db_path: str | Path | None = None,
) -> int:
    """Save a prediction and return its ID."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO predictions
               (squad_a_id, squad_b_id, win_prob, draw_prob, loss_prob,
                predicted_scoreline, model_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (squad_a_id, squad_b_id, win_prob, draw_prob, loss_prob,
             predicted_scoreline, model_version),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def log_real_result(
    prediction_id: int,
    squad_a_id: int,
    squad_b_id: int,
    actual_scoreline: str,
    actual_outcome: str,
    db_path: str | Path | None = None,
) -> int:
    """Log a real match result and return its ID."""
    conn = _get_connection(db_path)
    try:
        cursor = conn.execute(
            """INSERT INTO real_match_results
               (prediction_id, squad_a_id, squad_b_id, actual_scoreline, actual_outcome)
               VALUES (?, ?, ?, ?, ?)""",
            (prediction_id, squad_a_id, squad_b_id, actual_scoreline, actual_outcome),
        )
        conn.commit()
        return cursor.lastrowid
    finally:
        conn.close()


def get_squad(squad_id: int, db_path: str | Path | None = None) -> dict | None:
    """Get a squad by ID."""
    conn = _get_connection(db_path)
    try:
        row = conn.execute("SELECT * FROM squads WHERE id = ?", (squad_id,)).fetchone()
        if row is None:
            return None
        return dict(row)
    finally:
        conn.close()


def get_all_squads(db_path: str | Path | None = None) -> list[dict]:
    """Return all saved squads."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute("SELECT * FROM squads ORDER BY created_at DESC").fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_all_real_results(db_path: str | Path | None = None) -> list[dict]:
    """Return all logged real match results with squad and prediction details."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT r.*, 
                      p.win_prob, p.draw_prob, p.loss_prob, 
                      p.predicted_scoreline, p.model_version,
                      sa.name as squad_a_name, sa.formation as squad_a_formation,
                      sb.name as squad_b_name, sb.formation as squad_b_formation
               FROM real_match_results r
               LEFT JOIN predictions p ON r.prediction_id = p.id
               LEFT JOIN squads sa ON r.squad_a_id = sa.id
               LEFT JOIN squads sb ON r.squad_b_id = sb.id
               ORDER BY r.logged_at DESC"""
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_recent_predictions(
    limit: int = 20,
    db_path: str | Path | None = None,
) -> list[dict]:
    """Return the most recent predictions with squad names."""
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT p.*,
                      sa.name as squad_a_name, sa.formation as squad_a_formation,
                      sb.name as squad_b_name, sb.formation as squad_b_formation
               FROM predictions p
               LEFT JOIN squads sa ON p.squad_a_id = sa.id
               LEFT JOIN squads sb ON p.squad_b_id = sb.id
               ORDER BY p.created_at DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [dict(r) for r in rows]
    finally:
        conn.close()


def get_performance_stats(db_path: str | Path | None = None) -> dict:
    """Calculate prediction accuracy statistics.

    Returns:
        Dict with:
        - total_matches: number of logged real results
        - correct_predictions: number where predicted outcome matched actual
        - accuracy: correct / total (0.0 if no matches)
        - outcome_breakdown: dict of {outcome: {total, correct}} for win/draw/loss
    """
    conn = _get_connection(db_path)
    try:
        rows = conn.execute(
            """SELECT r.actual_outcome,
                      p.win_prob, p.draw_prob, p.loss_prob
               FROM real_match_results r
               JOIN predictions p ON r.prediction_id = p.id"""
        ).fetchall()

        if not rows:
            return {
                "total_matches": 0,
                "correct_predictions": 0,
                "accuracy": 0.0,
                "outcome_breakdown": {},
            }

        total = len(rows)
        correct = 0
        breakdown: dict[str, dict[str, int]] = {
            "win": {"total": 0, "correct": 0},
            "draw": {"total": 0, "correct": 0},
            "loss": {"total": 0, "correct": 0},
        }

        for row in rows:
            actual = row["actual_outcome"]
            probs = {
                "win": row["win_prob"],
                "draw": row["draw_prob"],
                "loss": row["loss_prob"],
            }
            predicted = max(probs, key=probs.get)

            if actual in breakdown:
                breakdown[actual]["total"] += 1
                if predicted == actual:
                    breakdown[actual]["correct"] += 1
                    correct += 1

        return {
            "total_matches": total,
            "correct_predictions": correct,
            "accuracy": round(correct / total, 4) if total > 0 else 0.0,
            "outcome_breakdown": breakdown,
        }
    finally:
        conn.close()
