"""Retraining loop — blend synthetic + real data and retrain the model."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from squadpredict.config import (
    CLASSIFIER_PATH,
    DB_PATH,
    MODELS_DIR,
    REAL_DATA_WEIGHT,
    SYNTHETIC_TRAINING_CSV,
)
from squadpredict.db import get_all_real_results, get_squad
from squadpredict.features import (
    FEATURE_NAMES,
    build_match_feature_vector,
    build_team_features,
)
from squadpredict.model import save_model, train


def _real_result_to_features(result: dict, db_path: str | Path | None = None) -> dict | None:
    """Convert a real match result row into a feature dict for training.

    Loads both squads' player data, builds features, computes the
    difference-based feature vector, and labels with the actual outcome.

    Returns None if squad data cannot be loaded/parsed.
    """
    try:
        squad_a = get_squad(result["squad_a_id"], db_path=db_path)
        squad_b = get_squad(result["squad_b_id"], db_path=db_path)

        if squad_a is None or squad_b is None:
            return None

        players_a = json.loads(squad_a["players_json"])
        players_b = json.loads(squad_b["players_json"])

        features_a = build_team_features(players_a, squad_a["formation"])
        features_b = build_team_features(players_b, squad_b["formation"])

        feature_vec = build_match_feature_vector(features_a, features_b)

        row = dict(zip(FEATURE_NAMES, feature_vec))
        row["outcome"] = result["actual_outcome"]
        return row

    except (json.JSONDecodeError, KeyError, TypeError) as e:
        print(f"Warning: Could not process real result {result.get('id', '?')}: {e}")
        return None


def blend_and_retrain(
    db_path: str | Path | None = None,
    synthetic_csv_path: str | Path | None = None,
    model_output_path: str | Path | None = None,
) -> str:
    """Retrain model with blended synthetic + real data.

    Real match results are weighted higher (REAL_DATA_WEIGHT) to shift
    the model toward real-world patterns while maintaining the synthetic
    data foundation.

    Args:
        db_path: Path to SQLite database. Defaults to config.DB_PATH.
        synthetic_csv_path: Path to synthetic training CSV.
        model_output_path: Where to save the new model.

    Returns:
        New model version string (timestamp-based).
    """
    db = db_path or DB_PATH
    synth_path = Path(synthetic_csv_path or SYNTHETIC_TRAINING_CSV)

    # Load synthetic data
    if not synth_path.exists():
        raise FileNotFoundError(
            f"Synthetic training data not found: {synth_path}. "
            "Run scripts/generate_training_data.py first."
        )

    print("Loading synthetic training data...")
    synthetic_df = pd.read_csv(synth_path)
    # Keep only feature columns + outcome
    synthetic_df = synthetic_df[FEATURE_NAMES + ["outcome"]]
    print(f"  Synthetic samples: {len(synthetic_df)}")

    # Load and convert real match results
    print("Loading real match results...")
    real_results = get_all_real_results(db_path=db)
    real_rows = []
    for result in real_results:
        row = _real_result_to_features(result, db_path=db)
        if row is not None:
            real_rows.append(row)

    print(f"  Real match samples: {len(real_rows)}")

    # Blend data
    if real_rows:
        real_df = pd.DataFrame(real_rows)
        blended_df = pd.concat([synthetic_df, real_df], ignore_index=True)

        # Build sample weights: 1.0 for synthetic, REAL_DATA_WEIGHT for real
        weights = np.ones(len(blended_df))
        weights[len(synthetic_df):] = REAL_DATA_WEIGHT
    else:
        blended_df = synthetic_df
        weights = None
        print("  No real data yet — retraining on synthetic data only.")

    print(f"  Total training samples: {len(blended_df)}")

    # Train
    print("\nTraining model...")
    clf, metrics = train(blended_df, sample_weights=weights)

    # Save with version
    version = datetime.now().strftime("v%Y%m%d_%H%M%S")
    output_path = Path(model_output_path) if model_output_path else CLASSIFIER_PATH

    # Save versioned copy
    versioned_path = MODELS_DIR / f"classifier_{version}.pkl"
    save_model(clf, versioned_path)

    # Save as latest
    save_model(clf, output_path)

    print(f"\nRetrained model version: {version}")
    print(f"  Accuracy: {metrics['accuracy']:.4f}")
    print(f"  Saved to: {output_path}")
    print(f"  Versioned copy: {versioned_path}")

    return version
