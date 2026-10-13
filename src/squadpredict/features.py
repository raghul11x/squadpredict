"""Feature engineering — convert squad data to model-ready feature vectors."""

from __future__ import annotations

from squadpredict.config import FORMATION_ATTACK_BIAS


# Position groupings for aggregation (includes both in-game and legacy labels)
POSITION_GROUPS: dict[str, list[str]] = {
    "GK": ["GK"],
    "DEF": ["CB", "LB", "RB", "LWB", "RWB"],
    "MID": ["CMF", "CM", "DMF", "CDM", "AMF", "CAM", "LMF", "LM", "RMF", "RM", "AM"],
    "ATK": ["ST", "CF", "LWF", "LW", "RWF", "RW", "SS"],
}


def _classify_position(position: str) -> str:
    """Map a specific position to its group (DEF/MID/ATK/GK)."""
    pos = position.strip().upper()
    for group, positions in POSITION_GROUPS.items():
        if pos in positions:
            return group
    # Default to MID for unknown positions
    return "MID"


def build_team_features(players: list[dict], formation: str) -> dict:
    """Aggregate player stats into a team feature vector.

    Args:
        players: List of player dicts, each with keys:
            position, overall_rating, pace, dribbling,
            attack_rating, defense_rating, chemistry_style
        formation: Formation string like "4-3-3"

    Returns:
        Dict with aggregated team features:
        - avg_overall: average overall rating across all players
        - avg_attack_rating: average attack rating of ATK-group players
        - avg_midfield_rating: average overall rating of MID-group players
        - avg_defense_rating: average defense rating of DEF-group players
        - avg_pace: average pace across all outfield players
        - avg_dribbling: average dribbling across all outfield players
        - avg_chemistry: average chemistry value (default 50 if not provided)
        - formation_attack_bias: from config lookup (default 0.5)
        - star_player_count: number of players with overall >= 90
        - squad_rating_variance: variance of overall ratings
    """
    if not players:
        return _empty_features(formation)

    # Classify players by position group
    grouped: dict[str, list[dict]] = {"GK": [], "DEF": [], "MID": [], "ATK": []}
    for p in players:
        group = _classify_position(p.get("position", "MID"))
        grouped[group].append(p)

    outfield = [p for p in players if _classify_position(p.get("position", "MID")) != "GK"]
    if not outfield:
        outfield = players  # fallback

    all_ratings = [p.get("overall_rating", 75) for p in players]

    # Attack rating: average attack_rating of ATK players, fallback to all outfield
    atk_players = grouped["ATK"] if grouped["ATK"] else outfield
    avg_attack = _safe_mean([p.get("attack_rating", 70) for p in atk_players])

    # Midfield rating: average overall of MID players, fallback to all outfield
    mid_players = grouped["MID"] if grouped["MID"] else outfield
    avg_midfield = _safe_mean([p.get("overall_rating", 75) for p in mid_players])

    # Defense rating: average defense_rating of DEF players, fallback to all outfield
    def_players = grouped["DEF"] if grouped["DEF"] else outfield
    avg_defense = _safe_mean([p.get("defense_rating", 70) for p in def_players])

    avg_pace = _safe_mean([p.get("pace", 70) for p in outfield])
    avg_dribbling = _safe_mean([p.get("dribbling", 70) for p in outfield])

    # Chemistry: default 50 if not provided
    chemistry_values = []
    for p in players:
        chem = p.get("chemistry", None)
        if chem is not None and isinstance(chem, (int, float)):
            chemistry_values.append(chem)
        else:
            chemistry_values.append(50)  # default
    avg_chemistry = _safe_mean(chemistry_values)

    formation_bias = FORMATION_ATTACK_BIAS.get(formation, 0.5)
    star_count = sum(1 for r in all_ratings if r >= 90)
    epic_count = sum(1 for p in players if "Epic" in p.get("card_type", ""))
    bigtime_count = sum(1 for p in players if p.get("card_type") == "Big Time")
    showtime_count = sum(1 for p in players if p.get("card_type") == "Show Time")

    # Variance
    mean_rating = _safe_mean(all_ratings)
    variance = _safe_mean([(r - mean_rating) ** 2 for r in all_ratings])

    return {
        "avg_overall": round(mean_rating, 2),
        "avg_attack_rating": round(avg_attack, 2),
        "avg_midfield_rating": round(avg_midfield, 2),
        "avg_defense_rating": round(avg_defense, 2),
        "avg_pace": round(avg_pace, 2),
        "avg_dribbling": round(avg_dribbling, 2),
        "avg_chemistry": round(avg_chemistry, 2),
        "formation_attack_bias": formation_bias,
        "star_player_count": star_count,
        "epic_count": epic_count,
        "bigtime_count": bigtime_count,
        "showtime_count": showtime_count,
        "squad_rating_variance": round(variance, 2),
    }


def _safe_mean(values: list[float | int]) -> float:
    """Return mean of values, defaulting to 0.0 for empty lists."""
    if not values:
        return 0.0
    return sum(values) / len(values)


def _empty_features(formation: str) -> dict:
    """Return zeroed feature dict for an empty squad."""
    return {
        "avg_overall": 0.0,
        "avg_attack_rating": 0.0,
        "avg_midfield_rating": 0.0,
        "avg_defense_rating": 0.0,
        "avg_pace": 0.0,
        "avg_dribbling": 0.0,
        "avg_chemistry": 50.0,
        "formation_attack_bias": FORMATION_ATTACK_BIAS.get(formation, 0.5),
        "star_player_count": 0,
        "epic_count": 0,
        "bigtime_count": 0,
        "showtime_count": 0,
        "squad_rating_variance": 0.0,
    }


def build_match_feature_vector(features_a: dict, features_b: dict) -> list[float]:
    """Build a difference-based feature vector for ML model input.

    Returns a list of floats representing the difference between team A and team B features.
    This makes the model symmetric: swapping teams just negates the features.
    """
    return [
        features_a["avg_attack_rating"] - features_b["avg_attack_rating"],
        features_a["avg_midfield_rating"] - features_b["avg_midfield_rating"],
        features_a["avg_defense_rating"] - features_b["avg_defense_rating"],
        features_a["avg_pace"] - features_b["avg_pace"],
        features_a["avg_dribbling"] - features_b["avg_dribbling"],
        features_a["avg_chemistry"] - features_b["avg_chemistry"],
        features_a["formation_attack_bias"],
        features_b["formation_attack_bias"],
        features_a["star_player_count"] - features_b["star_player_count"],
        features_a["squad_rating_variance"] - features_b["squad_rating_variance"],
    ]


FEATURE_NAMES: list[str] = [
    "diff_attack",
    "diff_midfield",
    "diff_defense",
    "diff_pace",
    "diff_dribbling",
    "diff_chemistry",
    "formation_bias_a",
    "formation_bias_b",
    "star_diff",
    "variance_diff",
]
