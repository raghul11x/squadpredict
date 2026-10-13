"""Generate synthetic training data using the Monte Carlo simulator."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from squadpredict.config import SYNTHETIC_DATA_COUNT, SYNTHETIC_TRAINING_CSV, KNOWN_FORMATIONS
from squadpredict.features import build_team_features, build_match_feature_vector, FEATURE_NAMES
from squadpredict.simulator import simulate_match


def _generate_random_squad(rng: np.random.Generator) -> tuple[list[dict], str]:
    """Generate a random but realistic squad with 11 players."""
    formation = rng.choice(KNOWN_FORMATIONS)

    # Parse formation to determine position distribution
    parts = [int(x) for x in formation.split("-")]

    positions_map = {
        2: ["CB", "CB"],
        3: ["CB", "CB", "CB"],
        4: ["LB", "CB", "CB", "RB"],
        5: ["LWB", "CB", "CB", "CB", "RWB"],
    }

    mid_positions_map = {
        1: ["CDM"],
        2: ["CM", "CM"],
        3: ["CM", "CAM", "CM"],
        4: ["LM", "CM", "CM", "RM"],
        5: ["LM", "CM", "CAM", "CM", "RM"],
    }

    atk_positions_map = {
        1: ["ST"],
        2: ["ST", "ST"],
        3: ["LW", "ST", "RW"],
        4: ["LW", "ST", "ST", "RW"],
    }

    # Build position list: GK + defense + midfield lines + attack
    positions = ["GK"]

    if len(parts) == 3:
        # e.g. 4-3-3
        def_count, mid_count, atk_count = parts
        positions += positions_map.get(def_count, ["CB"] * def_count)
        positions += mid_positions_map.get(mid_count, ["CM"] * mid_count)
        positions += atk_positions_map.get(atk_count, ["ST"] * atk_count)
    elif len(parts) == 4:
        # e.g. 4-2-3-1 -> defense, dmid, amid, attack
        def_count = parts[0]
        mid1 = parts[1]
        mid2 = parts[2]
        atk_count = parts[3]
        positions += positions_map.get(def_count, ["CB"] * def_count)
        positions += ["CDM"] * mid1
        positions += mid_positions_map.get(mid2, ["CAM"] * mid2)
        positions += atk_positions_map.get(atk_count, ["ST"] * atk_count)
    else:
        # Fallback: fill remaining positions with CM/ST
        positions += positions_map.get(parts[0], ["CB"] * parts[0])
        remaining = 10 - parts[0]
        positions += ["CM"] * (remaining // 2) + ["ST"] * (remaining - remaining // 2)

    # Ensure exactly 11 players
    while len(positions) < 11:
        positions.append("CM")
    positions = positions[:11]

    # Generate player stats
    # Team quality tier: determines base rating range
    team_quality = rng.choice(["weak", "average", "strong", "elite"], p=[0.15, 0.35, 0.35, 0.15])
    base_ranges = {
        "weak": (65, 78),
        "average": (74, 85),
        "strong": (80, 90),
        "elite": (86, 95),
    }
    base_low, base_high = base_ranges[team_quality]

    players = []
    for pos in positions:
        overall = int(rng.integers(base_low, base_high + 1))

        # Position-appropriate stat generation
        if pos == "GK":
            pace = int(rng.integers(40, 60))
            dribbling = int(rng.integers(40, 65))
            attack = int(rng.integers(35, 55))
            defense = int(rng.integers(overall - 5, min(overall + 3, 99)))
        elif pos in ["CB", "LB", "RB", "LWB", "RWB"]:
            pace = int(rng.integers(60, 92)) if pos in ["LB", "RB", "LWB", "RWB"] else int(rng.integers(55, 82))
            dribbling = int(rng.integers(55, 80))
            attack = int(rng.integers(45, 75))
            defense = int(rng.integers(overall - 8, min(overall + 2, 99)))
        elif pos in ["CDM", "CM", "CAM", "LM", "RM", "AM"]:
            pace = int(rng.integers(60, 88))
            dribbling = int(rng.integers(65, 95))
            attack = int(rng.integers(65, 90))
            defense = int(rng.integers(55, 82))
        else:  # ATK positions
            pace = int(rng.integers(75, 97))
            dribbling = int(rng.integers(75, 96))
            attack = int(rng.integers(overall - 5, min(overall + 3, 99)))
            defense = int(rng.integers(30, 55))

        players.append({
            "position": pos,
            "overall_rating": overall,
            "pace": pace,
            "dribbling": dribbling,
            "attack_rating": attack,
            "defense_rating": defense,
            "chemistry": int(rng.integers(30, 100)),
        })

    return players, formation


def main() -> None:
    """Generate synthetic squad pairs and simulate matches."""
    print(f"Generating {SYNTHETIC_DATA_COUNT} synthetic match samples...")

    rng = np.random.default_rng(42)
    rows: list[dict] = []

    for i in range(SYNTHETIC_DATA_COUNT):
        if (i + 1) % 1000 == 0:
            print(f"  Generated {i + 1}/{SYNTHETIC_DATA_COUNT}...")

        # Generate two random squads
        players_a, formation_a = _generate_random_squad(rng)
        players_b, formation_b = _generate_random_squad(rng)

        # Build features
        features_a = build_team_features(players_a, formation_a)
        features_b = build_team_features(players_b, formation_b)

        # Simulate match
        sim_result = simulate_match(features_a, features_b, n_sims=100, seed=int(rng.integers(0, 2**31)))

        # Build feature vector
        feature_vec = build_match_feature_vector(features_a, features_b)

        row = dict(zip(FEATURE_NAMES, feature_vec))
        # Sample outcome from the probability distribution rather than using
        # the majority outcome — this produces realistic draw rates (~20-25%)
        outcome_choices = ["win", "draw", "loss"]
        outcome_probs = [sim_result["win_prob"], sim_result["draw_prob"], sim_result["loss_prob"]]
        row["outcome"] = rng.choice(outcome_choices, p=outcome_probs)
        row["score_a"] = sim_result["avg_goals_a"]
        row["score_b"] = sim_result["avg_goals_b"]
        rows.append(row)

    df = pd.DataFrame(rows)

    # Ensure output directory exists
    SYNTHETIC_TRAINING_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SYNTHETIC_TRAINING_CSV, index=False)

    print(f"\nSaved {len(df)} samples to {SYNTHETIC_TRAINING_CSV}")
    print(f"\nOutcome distribution:")
    print(df["outcome"].value_counts(normalize=True).round(4))
    print(f"\nFeature statistics:")
    print(df[FEATURE_NAMES].describe().round(2))


if __name__ == "__main__":
    main()
