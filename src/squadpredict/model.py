"""ML model — train and predict match outcomes."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

from squadpredict.config import CLASSIFIER_PATH
from squadpredict.features import build_match_feature_vector, FEATURE_NAMES
from squadpredict.simulator import simulate_match


def train(
    df: pd.DataFrame,
    sample_weights: np.ndarray | None = None,
    test_size: float = 0.2,
    random_state: int = 42,
) -> tuple[GradientBoostingClassifier, dict]:
    """Train a GradientBoostingClassifier on match feature data.

    Args:
        df: DataFrame with columns matching FEATURE_NAMES + 'outcome'.
            'outcome' should be one of: 'win', 'draw', 'loss'.
        sample_weights: Optional per-sample weights for training.
        test_size: Fraction of data to hold out for evaluation.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (trained classifier, metrics dict with accuracy and report).
    """
    X = df[FEATURE_NAMES].to_numpy()
    y = df["outcome"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y,
    )

    # Split sample weights if provided
    if sample_weights is not None:
        w_train, w_test = train_test_split(
            sample_weights, test_size=test_size, random_state=random_state, stratify=y,
        )
    else:
        w_train = None

    clf = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=4,
        min_samples_split=10,
        min_samples_leaf=5,
        subsample=0.8,
        random_state=random_state,
    )

    clf.fit(X_train, y_train, sample_weight=w_train)

    y_pred = clf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)

    metrics = {
        "accuracy": round(accuracy, 4),
        "classification_report": report,
        "test_size": len(y_test),
        "train_size": len(y_train),
    }

    print(f"Model accuracy: {accuracy:.4f}")
    print(f"Train size: {len(y_train)}, Test size: {len(y_test)}")
    print(classification_report(y_test, y_pred))

    return clf, metrics


def predict(
    features_a: dict,
    features_b: dict,
    model: GradientBoostingClassifier,
) -> dict:
    """Predict match outcome probabilities and scoreline.

    Args:
        features_a: Team A feature dict from build_team_features().
        features_b: Team B feature dict from build_team_features().
        model: Trained GradientBoostingClassifier.

    Returns:
        Dict with:
        - win_prob: P(team A wins)
        - draw_prob: P(draw)
        - loss_prob: P(team A loses)
        - predicted_scoreline: Estimated scoreline from simulator
        - predicted_outcome: Most likely outcome
    """
    feature_vec = build_match_feature_vector(features_a, features_b)
    X = np.array([feature_vec])

    # Get class probabilities
    probas = model.predict_proba(X)[0]
    classes = model.classes_

    # Map class names to probabilities
    prob_dict: dict[str, float] = {}
    for cls, prob in zip(classes, probas):
        prob_dict[cls] = round(float(prob), 4)

    win_prob = prob_dict.get("win", 0.0)
    draw_prob = prob_dict.get("draw", 0.0)
    loss_prob = prob_dict.get("loss", 0.0)

    # Use simulator for scoreline prediction (quick run with fewer sims)
    sim_result = simulate_match(features_a, features_b, n_sims=500)

    predicted_outcome = max(prob_dict, key=prob_dict.get)

    return {
        "win_prob": win_prob,
        "draw_prob": draw_prob,
        "loss_prob": loss_prob,
        "predicted_scoreline": sim_result["predicted_scoreline"],
        "predicted_outcome": predicted_outcome,
    }


def save_model(model: GradientBoostingClassifier, path: str | Path | None = None) -> Path:
    """Save trained model to disk."""
    save_path = Path(path) if path else CLASSIFIER_PATH
    save_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, save_path)
    print(f"Model saved to {save_path}")
    return save_path


def load_model(path: str | Path | None = None) -> GradientBoostingClassifier:
    """Load trained model from disk."""
    load_path = Path(path) if path else CLASSIFIER_PATH
    if not load_path.exists():
        raise FileNotFoundError(f"Model not found: {load_path}")
    model = joblib.load(load_path)
    print(f"Model loaded from {load_path}")
    return model
