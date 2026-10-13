"""Tests for ML model training and prediction."""

import tempfile
from pathlib import Path

import pandas as pd
import pytest

from squadpredict.features import (
    FEATURE_NAMES,
    build_match_feature_vector,
    build_team_features,
)
from squadpredict.model import load_model, predict, save_model, train
from squadpredict.config import CLASSIFIER_PATH


# --- Fixtures ---

@pytest.fixture
def sample_training_df() -> pd.DataFrame:
    """Create a small synthetic training DataFrame."""
    import numpy as np
    from squadpredict.simulator import simulate_match

    rng = np.random.default_rng(42)
    rows = []
    for _ in range(200):
        features_a = {
            "avg_overall": rng.uniform(65, 95),
            "avg_attack_rating": rng.uniform(60, 95),
            "avg_midfield_rating": rng.uniform(60, 95),
            "avg_defense_rating": rng.uniform(60, 95),
            "avg_pace": rng.uniform(60, 95),
            "avg_dribbling": rng.uniform(60, 95),
            "avg_chemistry": rng.uniform(30, 100),
            "formation_attack_bias": rng.uniform(0.3, 0.85),
            "star_player_count": int(rng.integers(0, 8)),
            "squad_rating_variance": rng.uniform(0, 20),
        }
        features_b = {
            "avg_overall": rng.uniform(65, 95),
            "avg_attack_rating": rng.uniform(60, 95),
            "avg_midfield_rating": rng.uniform(60, 95),
            "avg_defense_rating": rng.uniform(60, 95),
            "avg_pace": rng.uniform(60, 95),
            "avg_dribbling": rng.uniform(60, 95),
            "avg_chemistry": rng.uniform(30, 100),
            "formation_attack_bias": rng.uniform(0.3, 0.85),
            "star_player_count": int(rng.integers(0, 8)),
            "squad_rating_variance": rng.uniform(0, 20),
        }

        sim = simulate_match(features_a, features_b, n_sims=50, seed=int(rng.integers(0, 2**31)))
        vec = build_match_feature_vector(features_a, features_b)
        row = dict(zip(FEATURE_NAMES, vec))
        outcome_choices = ["win", "draw", "loss"]
        outcome_probs = [sim["win_prob"], sim["draw_prob"], sim["loss_prob"]]
        row["outcome"] = rng.choice(outcome_choices, p=outcome_probs)
        rows.append(row)

    return pd.DataFrame(rows)


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


# --- Tests ---

class TestTrain:
    def test_returns_classifier_and_metrics(self, sample_training_df):
        clf, metrics = train(sample_training_df, test_size=0.3)
        assert clf is not None
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_classifier_has_predict_proba(self, sample_training_df):
        clf, _ = train(sample_training_df)
        assert hasattr(clf, "predict_proba")

    def test_classifier_classes(self, sample_training_df):
        clf, _ = train(sample_training_df)
        classes = set(clf.classes_)
        # Should contain at least win and loss (draw may be absent in small samples)
        assert "win" in classes or "loss" in classes


class TestPredict:
    def test_returns_all_keys(self, sample_training_df, strong_features, weak_features):
        clf, _ = train(sample_training_df)
        result = predict(strong_features, weak_features, clf)
        assert "win_prob" in result
        assert "draw_prob" in result
        assert "loss_prob" in result
        assert "predicted_scoreline" in result
        assert "predicted_outcome" in result

    def test_probabilities_sum_to_one(self, sample_training_df, strong_features, weak_features):
        clf, _ = train(sample_training_df)
        result = predict(strong_features, weak_features, clf)
        total = result["win_prob"] + result["draw_prob"] + result["loss_prob"]
        assert abs(total - 1.0) < 0.01

    def test_probabilities_are_valid(self, sample_training_df, strong_features, weak_features):
        clf, _ = train(sample_training_df)
        result = predict(strong_features, weak_features, clf)
        for key in ["win_prob", "draw_prob", "loss_prob"]:
            assert 0.0 <= result[key] <= 1.0


class TestSaveLoadModel:
    def test_save_and_load_roundtrip(self, sample_training_df, tmp_path):
        clf, _ = train(sample_training_df)
        model_path = tmp_path / "test_model.pkl"
        save_model(clf, model_path)
        assert model_path.exists()

        loaded = load_model(model_path)
        assert hasattr(loaded, "predict_proba")

    def test_load_nonexistent_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_model(tmp_path / "nonexistent.pkl")
