"""Tests for feature engineering module."""

import pytest

from squadpredict.features import (
    FEATURE_NAMES,
    _classify_position,
    _safe_mean,
    build_match_feature_vector,
    build_team_features,
)


# --- Fixtures ---

@pytest.fixture
def sample_players() -> list[dict]:
    """A realistic 11-player squad."""
    return [
        {"position": "GK", "overall_rating": 90, "pace": 50, "dribbling": 55, "attack_rating": 50, "defense_rating": 92, "chemistry": 85},
        {"position": "CB", "overall_rating": 90, "pace": 78, "dribbling": 72, "attack_rating": 60, "defense_rating": 93, "chemistry": 90},
        {"position": "CB", "overall_rating": 89, "pace": 74, "dribbling": 68, "attack_rating": 55, "defense_rating": 92, "chemistry": 85},
        {"position": "LB", "overall_rating": 87, "pace": 95, "dribbling": 84, "attack_rating": 78, "defense_rating": 75, "chemistry": 80},
        {"position": "RB", "overall_rating": 87, "pace": 93, "dribbling": 82, "attack_rating": 80, "defense_rating": 76, "chemistry": 80},
        {"position": "CM", "overall_rating": 91, "pace": 74, "dribbling": 87, "attack_rating": 88, "defense_rating": 65, "chemistry": 90},
        {"position": "CM", "overall_rating": 90, "pace": 82, "dribbling": 86, "attack_rating": 85, "defense_rating": 80, "chemistry": 85},
        {"position": "CAM", "overall_rating": 88, "pace": 78, "dribbling": 90, "attack_rating": 82, "defense_rating": 70, "chemistry": 80},
        {"position": "LW", "overall_rating": 90, "pace": 95, "dribbling": 94, "attack_rating": 86, "defense_rating": 40, "chemistry": 85},
        {"position": "RW", "overall_rating": 90, "pace": 90, "dribbling": 90, "attack_rating": 90, "defense_rating": 45, "chemistry": 85},
        {"position": "ST", "overall_rating": 91, "pace": 89, "dribbling": 80, "attack_rating": 93, "defense_rating": 45, "chemistry": 90},
    ]


@pytest.fixture
def weak_players() -> list[dict]:
    """A weak 11-player squad."""
    return [
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


# --- Tests ---

class TestClassifyPosition:
    def test_gk(self):
        assert _classify_position("GK") == "GK"

    def test_defenders(self):
        for pos in ["CB", "LB", "RB", "LWB", "RWB"]:
            assert _classify_position(pos) == "DEF"

    def test_midfielders(self):
        for pos in ["CMF", "CM", "DMF", "CDM", "AMF", "CAM", "LMF", "LM", "RMF", "RM"]:
            assert _classify_position(pos) == "MID"

    def test_attackers(self):
        for pos in ["ST", "CF", "LWF", "LW", "RWF", "RW", "SS"]:
            assert _classify_position(pos) == "ATK"

    def test_unknown_defaults_to_mid(self):
        assert _classify_position("XYZ") == "MID"

    def test_case_insensitive(self):
        assert _classify_position("gk") == "GK"
        assert _classify_position("Cb") == "DEF"


class TestSafeMean:
    def test_normal_list(self):
        assert _safe_mean([1, 2, 3]) == 2.0

    def test_empty_list(self):
        assert _safe_mean([]) == 0.0

    def test_single_value(self):
        assert _safe_mean([42]) == 42.0


class TestBuildTeamFeatures:
    def test_returns_all_keys(self, sample_players):
        features = build_team_features(sample_players, "4-3-3")
        expected_keys = {
            "avg_overall", "avg_attack_rating", "avg_midfield_rating",
            "avg_defense_rating", "avg_pace", "avg_dribbling",
            "avg_chemistry", "formation_attack_bias", "star_player_count",
            "epic_count", "bigtime_count", "showtime_count",
            "squad_rating_variance",
        }
        assert set(features.keys()) == expected_keys

    def test_values_in_sane_ranges(self, sample_players):
        features = build_team_features(sample_players, "4-3-3")
        assert 60 <= features["avg_overall"] <= 99
        assert 40 <= features["avg_attack_rating"] <= 99
        assert 40 <= features["avg_defense_rating"] <= 99
        assert 40 <= features["avg_pace"] <= 99
        assert 0 <= features["formation_attack_bias"] <= 1.0
        assert features["star_player_count"] >= 0
        assert features["squad_rating_variance"] >= 0

    def test_strong_squad_higher_ratings(self, sample_players, weak_players):
        strong = build_team_features(sample_players, "4-3-3")
        weak = build_team_features(weak_players, "4-3-3")
        assert strong["avg_overall"] > weak["avg_overall"]
        assert strong["avg_attack_rating"] > weak["avg_attack_rating"]
        assert strong["star_player_count"] > weak["star_player_count"]

    def test_empty_players(self):
        features = build_team_features([], "4-3-3")
        assert features["avg_overall"] == 0.0
        assert features["star_player_count"] == 0

    def test_formation_bias_lookup(self, sample_players):
        f1 = build_team_features(sample_players, "4-3-3")
        f2 = build_team_features(sample_players, "5-4-1")
        assert f1["formation_attack_bias"] > f2["formation_attack_bias"]


class TestBuildMatchFeatureVector:
    def test_returns_correct_length(self, sample_players, weak_players):
        fa = build_team_features(sample_players, "4-3-3")
        fb = build_team_features(weak_players, "4-4-2")
        vec = build_match_feature_vector(fa, fb)
        assert len(vec) == len(FEATURE_NAMES)

    def test_symmetric_negation(self, sample_players, weak_players):
        fa = build_team_features(sample_players, "4-3-3")
        fb = build_team_features(weak_players, "4-3-3")
        vec_ab = build_match_feature_vector(fa, fb)
        vec_ba = build_match_feature_vector(fb, fa)
        # Difference features should be negated (except formation biases which swap)
        for i in [0, 1, 2, 3, 4, 5, 8, 9]:  # diff features
            assert abs(vec_ab[i] + vec_ba[i]) < 0.01

    def test_all_values_are_float(self, sample_players, weak_players):
        fa = build_team_features(sample_players, "4-3-3")
        fb = build_team_features(weak_players, "4-3-3")
        vec = build_match_feature_vector(fa, fb)
        assert all(isinstance(v, (int, float)) for v in vec)
