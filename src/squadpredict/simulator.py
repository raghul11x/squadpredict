"""Monte Carlo match simulator — Poisson-based expected goals model."""

from __future__ import annotations

from collections import Counter

import numpy as np

from squadpredict.config import (
    BASE_GOAL_RATE,
    CHEMISTRY_BONUS_SCALE,
    DEFAULT_N_SIMS,
    FORMATION_ATTACK_BIAS,
    UPSET_FACTOR_STD,
)


def _compute_expected_goals(
    attack_rating: float,
    opponent_defense_rating: float,
    chemistry: float,
    formation_bias: float,
    opponent_formation_bias: float,
) -> float:
    """Compute expected goals (lambda) for a team.

    λ = (attack / defense) * base_rate * chemistry_bonus * formation_matchup

    Args:
        attack_rating: Team's average attack rating (typically 60-95).
        opponent_defense_rating: Opponent's average defense rating.
        chemistry: Team's average chemistry (0-100, default ~50).
        formation_bias: Team's formation attack bias (0-1).
        opponent_formation_bias: Opponent's formation attack bias (0-1).

    Returns:
        Expected goals (lambda) for Poisson sampling.
    """
    # Prevent division by zero or extremely low defense
    defense = max(opponent_defense_rating, 40.0)
    attack = max(attack_rating, 40.0)

    # Base expected goals from attack vs defense ratio
    ratio = attack / defense
    lambda_base = ratio * BASE_GOAL_RATE

    # Chemistry bonus: scales by how much chemistry deviates from neutral (50)
    chemistry_bonus = 1.0 + CHEMISTRY_BONUS_SCALE * (chemistry - 50.0) / 50.0

    # Formation matchup: attacking formation vs defensive opponent = slight advantage
    # Wide/attacking formation vs narrow/defensive = bonus
    formation_matchup = 1.0 + 0.1 * (formation_bias - opponent_formation_bias)

    lambda_val = lambda_base * chemistry_bonus * formation_matchup

    # Clamp to reasonable range [0.2, 4.0] goals expected
    return max(0.2, min(4.0, lambda_val))


def simulate_match(
    features_a: dict,
    features_b: dict,
    n_sims: int = DEFAULT_N_SIMS,
    seed: int | None = None,
) -> dict:
    """Run Monte Carlo match simulations and return outcome probabilities.

    Each simulation:
    1. Compute expected goals (lambda) for each team.
    2. Add upset noise to lambda.
    3. Draw goals from Poisson(lambda).
    4. Determine outcome (win/draw/loss from team A's perspective).

    Args:
        features_a: Team A feature dict from build_team_features().
        features_b: Team B feature dict from build_team_features().
        n_sims: Number of Monte Carlo simulations to run.
        seed: Optional random seed for reproducibility.

    Returns:
        Dict with:
        - win_prob: P(team A wins)
        - draw_prob: P(draw)
        - loss_prob: P(team A loses)
        - avg_goals_a: Average goals scored by team A
        - avg_goals_b: Average goals scored by team B
        - predicted_scoreline: Most common scoreline (e.g. "2-1")
    """
    rng = np.random.default_rng(seed)

    # Compute base expected goals
    lambda_a = _compute_expected_goals(
        attack_rating=features_a["avg_attack_rating"],
        opponent_defense_rating=features_b["avg_defense_rating"],
        chemistry=features_a["avg_chemistry"],
        formation_bias=features_a["formation_attack_bias"],
        opponent_formation_bias=features_b["formation_attack_bias"],
    )
    lambda_b = _compute_expected_goals(
        attack_rating=features_b["avg_attack_rating"],
        opponent_defense_rating=features_a["avg_defense_rating"],
        chemistry=features_b["avg_chemistry"],
        formation_bias=features_b["formation_attack_bias"],
        opponent_formation_bias=features_a["formation_attack_bias"],
    )

    wins = 0
    draws = 0
    losses = 0
    total_goals_a = 0
    total_goals_b = 0
    scorelines: list[tuple[int, int]] = []

    for _ in range(n_sims):
        # Add upset factor noise
        noise_a = rng.normal(0, UPSET_FACTOR_STD)
        noise_b = rng.normal(0, UPSET_FACTOR_STD)

        lam_a = max(0.1, lambda_a + noise_a)
        lam_b = max(0.1, lambda_b + noise_b)

        # Draw goals from Poisson distribution
        goals_a = int(rng.poisson(lam_a))
        goals_b = int(rng.poisson(lam_b))

        total_goals_a += goals_a
        total_goals_b += goals_b
        scorelines.append((goals_a, goals_b))

        if goals_a > goals_b:
            wins += 1
        elif goals_a < goals_b:
            losses += 1
        else:
            draws += 1

    # Most common scoreline
    scoreline_counts = Counter(scorelines)
    most_common_score = scoreline_counts.most_common(1)[0][0]

    return {
        "win_prob": round(wins / n_sims, 4),
        "draw_prob": round(draws / n_sims, 4),
        "loss_prob": round(losses / n_sims, 4),
        "avg_goals_a": round(total_goals_a / n_sims, 2),
        "avg_goals_b": round(total_goals_b / n_sims, 2),
        "predicted_scoreline": f"{most_common_score[0]}-{most_common_score[1]}",
        "outcome": (
            "win" if wins > losses and wins > draws
            else "draw" if draws >= wins and draws >= losses
            else "loss"
        ),
    }
