"""Reference player database — load and fuzzy-match player names."""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
from rapidfuzz import fuzz, process

from squadpredict.config import PLAYER_MATCH_THRESHOLD, PLAYER_REFERENCE_CSV


def _normalize_name(name: str) -> str:
    """Normalize name by stripping accents and single-letter initial prefixes."""
    s = unicodedata.normalize("NFKD", name).encode("ASCII", "ignore").decode("utf-8")
    return re.sub(r"^[A-Z]\.\s*", "", s, flags=re.IGNORECASE).strip().lower()


def load_reference(csv_path: str | Path | None = None) -> pd.DataFrame:
    """Load the player reference CSV into a DataFrame.
    
    Args:
        csv_path: Path to the CSV file. Defaults to config.PLAYER_REFERENCE_CSV.
    
    Returns:
        DataFrame with player reference data.
    
    Raises:
        FileNotFoundError: If the CSV file doesn't exist.
    """
    path = Path(csv_path) if csv_path else Path(PLAYER_REFERENCE_CSV)
    if not path.exists():
        raise FileNotFoundError(f"Player reference CSV not found: {path}")
    df = pd.read_csv(path, encoding="utf-8")
    # Normalize column names
    df.columns = df.columns.str.strip().str.lower().str.replace(" ", "_")
    return df


def lookup_player(
    name: str,
    df: pd.DataFrame,
    threshold: int = PLAYER_MATCH_THRESHOLD,
) -> dict | None:
    """Fuzzy-match a player name against the reference database.
    
    Handles OCR typos, accent variations, and optional initial prefixes
    (e.g. "Mbape" -> "K. Mbappé", "Halland" -> "E. Haaland", "Messi" -> "L. Messi").
    
    Args:
        name: The name to search for (possibly OCR-garbled).
        df: The reference DataFrame from load_reference().
        threshold: Minimum fuzzy match score (0-100). Returns None below this.
    
    Returns:
        Dict with player data and match confidence, or None if no match above threshold.
    """
    if not name or not name.strip():
        return None
    
    name = name.strip()
    norm_query = _normalize_name(name)
    
    best_player = None
    best_score = 0.0
    
    for _, row in df.iterrows():
        orig_name = str(row["player_name"])
        norm_choice = _normalize_name(orig_name)
        
        # Calculate multiple similarity metrics and take the best
        score = max(
            fuzz.token_sort_ratio(name, orig_name),
            fuzz.ratio(norm_query, norm_choice),
            fuzz.token_set_ratio(norm_query, norm_choice),
        )
        
        if score > best_score:
            best_score = score
            best_player = row
            
    if best_score < threshold or best_player is None:
        return None
        
    player_data = best_player.to_dict()
    player_data["match_confidence"] = int(best_score)
    return player_data
