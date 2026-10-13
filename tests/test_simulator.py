"""Tests for Monte Carlo match simulator."""

import pytest

from squadpredict.features import build_team_features
from squadpredict.simulator import simulate_match, _compute_expected_goals


# --- Fixtures ---

@pytest.fixture
def strong_features() -> dict:
    players = [
        {"position": "GK", "overall_rating": 92, "pace": 50, "dribbling": 55, "attack_rating": 50, "defense_rating": 92, "chemistry": 90},
        {"position": "CB", "overall_rating": 91, "pace": 78, "dribbling": 72, "attack_rating": 60, "defense_rating": 93, "chemistry": 90},
        {"position": "CB", "overall_rating": 90, "pace": 80, "dribbling": 70, "attack_rating": 55, "defense_rating": 92, "chemistry": 90},
        {"position": "LB", "overall_rating": 89, "pace": 92, "dribbling": 82, "attack_rating": 78, "defense_rating": 82, "chemistry": 85},
        {"position": "RB", "overall_rating": 89, "pace": 90, "dribbling": 84, "attack_rating": 80, "defense_rating": 80, "chemistry": 85},
        {"position": "CM", "overall_rating": 91, "pace": 76, "dribbling": 88, "attack_rating": 88, "defense_rating": 70, "chemistry": 90},
        {"position": "CM", "overall_rating": 90, "pace": 80, "dribbling": 86, "attack_rating": 85, "defense_rating": 78, "chemistry": 90},
        {"position": "CAM", "overall_rating": 91, "pace": 82, "dribbling": 92, "attack_rating": 90, "defense_rating": 55, "chemistry": 85},
        {"position": "LW", "overall_rating": 92, "pace": 96, "dribbling": 94, "attack_rating": 88, "defense_rating": 40, "chemistry": 90},
        {"position": "RW", "overall_rating": 91, "pace": 90, "dribbling": 92, "attack_rating": 90, "defense_rating": 42, "chemistry": 85},
        {"position": "ST", "overall_rating": 93, "pace": 95, "dribbling": 90, "attack_rating": 93, "defense_rating": 40, "chemistry": 95},
    ]
    return build_team_features(players, "4-3-3")


@pytest.fixture
def weak_features() -> dict:
    players = [
        {"position": "GK", "overall_rating": 70, "pace": 45, "dribbling": 50, "attack_rating": 40, "defense_rating": 70, "chemistry": 40},
        {"position": "CB", "overall_rating": 68, "pace": 60, "dribbling": 55, "attack_rating": 45, "defense_rating": 68, "chemistry": 35},
        {"position": "CB", "overall_rating": 67, "pace": 58, "dribbling": 52, "attack_rating": 42, "defense_rating": 66, "chemistry": 35},
        {"position": "LB", "overall_rating": 66, "pace": 72, "dribbling": 60, "attack_rating": 55, "defense_rating": 62, "chemistry": 40},
        {"position": "RB", "overall_rating": 65, "pace": 70, "dribbling": 58, "attack_rating": 52, "defense_rating": 60, "chemistry": 40},
        {"position": "CM", "overall_rating": 68, "pace": 62, "dribbling": 65, "attack_rating": 65, "defense_rating": 58, "chemistry": 35},
        {"position": "CM", "overall_rating": 66, "pace": 60, "dribbling": 62, "attack_rating": 62, "defense_rating": 56, "chemistry": 35},
        {"position": "CAM", "overall_rating": 67, "pace": 65, "dribbling": 68, "attack_rating": 68, "defense_rating": 45, "chemistry": 40},
        {"position": "LW", "overall_rating": 66, "pace": 78, "dribbling": 70, "attack_rating": 66, "defense_rating": 35, "chemistry": 35},
        {"position": "RW", "overall_rating": 65, "pace": 76, "dribbling": 68, "attack_rating": 64, "defense_rating": 33, "chemistry": 35},
        {"position": "ST", "overall_rating": 68, "pace": 80, "dribbling": 72, "attack_rating": 70, "defense_rating": 32, "chemistry": 40},
    ]
    return build_team_features(players, "4-3-3")


@pytest.fixture
def equal_features() -> dict:
    players = [
        {"position": "GK", "overall_rating": 80, "pace": 50, "dribbling": 55, "attack_rating": 50, "defense_rating": 80, "chemistry": 60},
        {"position": "CB", "overall_rating": 80, "pace": 70, "dribbling": 65, "attack_rating": 55, "defense_rating": 80, "chemistry": 60},
        {"position": "CB", "overall_rating": 80, "pace": 68, "dribbling": 62, "attack_rating": 52, "defense_rating": 80, "chemistry": 60},
        {"position": "LB", "overall_rating": 80, "pace": 82, "dribbling": 72, "attack_rating": 68, "defense_rating": 72, "chemistry": 60},
        {"position": "RB", "overall_rating": 80, "pace": 80, "dribbling": 70, "attack_rating": 65, "defense_rating": 70, "chemistry": 60},
        {"position": "CM", "overall_rating": 80, "pace": 70, "dribbling": 78, "attack_rating": 78, "defense_rating": 65, "chemistry": 60},
        {"position": "CM", "overall_rating": 80, "pace": 72, "dribbling": 76, "attack_rating": 75, "defense_rating": 68, "chemistry": 60},
        {"position": "CAM", "overall_rating": 80, "pace": 75, "dribbling": 82, "attack_rating": 80, "defense_rating": 55, "chemistry": 60},
        {"position": "LW", "overall_rating": 80, "pace": 88, "dribbling": 85, "attack_rating": 78, "defense_rating": 38, "chemistry": 60},
        {"position": "RW", "overall_rating": 80, "pace": 86, "dribbling": 83, "attack_rating": 76, "defense_rating": 36, "chemistry": 60},
        {"position": "ST", "overall_rating": 80, "pace": 85, "dribbling": 80, "attack_rating": 82, "defense_rating": 35, "chemistry": 60},
    ]
    return build_team_features(players, "4-3-3")


# --- Tests ---

class TestComputeExpectedGoals:
    def test_returns_positive(self):
        result = _compute_expected_goals(85, 80, 50, 0.7, 0.5)
        assert result > 0

    def test_clamped_range(self):
        # Very high attack vs low defense
        result = _compute_expected_goals(99, 40, 100, 1.0, 0.0)
        assert result <= 4.0
        # Very low attack vs high defense
        result = _compute_expected_goals(40, 99, 0, 0.0, 1.0)
        assert result >= 0.2


class TestSimulateMatch:
    def test_probabilities_sum_to_one(self, strong_features, weak_features):
        result = simulate_match(strong_features, weak_features, n_sims=1000, seed=42)
        total = result["win_prob"] + result["draw_prob"] + result["loss_prob"]
        assert abs(total - 1.0) < 0.01

    def test_returns_all_keys(self, strong_features, weak_features):
        result = simulate_match(strong_features, weak_features, n_sims=100, seed=42)
        expected_keys = {
            "win_prob", "draw_prob", "loss_prob",
            "avg_goals_a", "avg_goals_b",
            "predicted_scoreline", "outcome",
        }
        assert set(result.keys()) == expected_keys

    def test_strong_team_favored(self, strong_features, weak_features):
        result = simulate_match(strong_features, weak_features, n_sims=2000, seed=42)
        assert result["win_prob"] > result["loss_prob"]

    def test_equal_teams_balanced(self, equal_features):
        result = simulate_match(equal_features, equal_features, n_sims=5000, seed=42)
        # For equal teams, win and loss should be roughly equal
        assert abs(result["win_prob"] - result["loss_prob"]) < 0.1

    def test_deterministic_with_seed(self, strong_features, weak_features):
        r1 = simulate_match(strong_features, weak_features, n_sims=100, seed=123)
        r2 = simulate_match(strong_features, weak_features, n_sims=100, seed=123)
        assert r1 == r2

    def test_scoreline_format(self, strong_features, weak_features):
        result = simulate_match(strong_features, weak_features, n_sims=100, seed=42)
        parts = result["predicted_scoreline"].split("-")
        assert len(parts) == 2
        assert all(p.isdigit() for p in parts)

    def test_avg_goals_positive(self, strong_features, weak_features):
        result = simulate_match(strong_features, weak_features, n_sims=1000, seed=42)
        assert result["avg_goals_a"] > 0
        assert result["avg_goals_b"] > 0
