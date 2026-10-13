"""Train the initial model on synthetic data."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Add project root to path for imports
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "src"))

from squadpredict.config import SYNTHETIC_TRAINING_CSV, CLASSIFIER_PATH
from squadpredict.features import FEATURE_NAMES
from squadpredict.model import train, save_model


def main() -> None:
    """Train v1 model on synthetic data and save to models/classifier.pkl."""
    print("Loading synthetic training data...")

    if not SYNTHETIC_TRAINING_CSV.exists():
        print(f"ERROR: Training data not found at {SYNTHETIC_TRAINING_CSV}")
        print("Run 'python scripts/generate_training_data.py' first.")
        sys.exit(1)

    df = pd.read_csv(SYNTHETIC_TRAINING_CSV)
    print(f"Loaded {len(df)} samples")
    print(f"Outcome distribution:\n{df['outcome'].value_counts()}\n")

    # Verify all required feature columns exist
    missing_cols = [c for c in FEATURE_NAMES if c not in df.columns]
    if missing_cols:
        print(f"ERROR: Missing feature columns: {missing_cols}")
        sys.exit(1)

    # Verify outcome column has expected values
    valid_outcomes = {"win", "draw", "loss"}
    actual_outcomes = set(df["outcome"].unique())
    if not actual_outcomes.issubset(valid_outcomes):
        print(f"WARNING: Unexpected outcome values: {actual_outcomes - valid_outcomes}")

    print("Training model...")
    clf, metrics = train(df)

    print(f"\nSaving model to {CLASSIFIER_PATH}...")
    save_model(clf, CLASSIFIER_PATH)

    print("\n=== Training Complete ===")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"Train/Test split: {metrics['train_size']}/{metrics['test_size']}")

    # Quick sanity check: predict on a clearly mismatched pair
    print("\n=== Sanity Check ===")
    from squadpredict.features import build_team_features, build_match_feature_vector
    from squadpredict.model import predict

    # Strong team
    strong_players = [
        {"position": "GK", "overall_rating": 92, "pace": 50, "dribbling": 55, "attack_rating": 50, "defense_rating": 92, "chemistry": 95},
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

    # Weak team
    weak_players = [
        {"position": "GK", "overall_rating": 72, "pace": 45, "dribbling": 50, "attack_rating": 40, "defense_rating": 72, "chemistry": 40},
        {"position": "CB", "overall_rating": 70, "pace": 60, "dribbling": 55, "attack_rating": 45, "defense_rating": 70, "chemistry": 35},
        {"position": "CB", "overall_rating": 69, "pace": 58, "dribbling": 52, "attack_rating": 42, "defense_rating": 68, "chemistry": 35},
        {"position": "LB", "overall_rating": 68, "pace": 72, "dribbling": 60, "attack_rating": 55, "defense_rating": 65, "chemistry": 40},
        {"position": "RB", "overall_rating": 67, "pace": 70, "dribbling": 58, "attack_rating": 52, "defense_rating": 64, "chemistry": 40},
        {"position": "CM", "overall_rating": 70, "pace": 62, "dribbling": 65, "attack_rating": 65, "defense_rating": 60, "chemistry": 35},
        {"position": "CM", "overall_rating": 68, "pace": 60, "dribbling": 62, "attack_rating": 62, "defense_rating": 58, "chemistry": 35},
        {"position": "CAM", "overall_rating": 69, "pace": 65, "dribbling": 68, "attack_rating": 68, "defense_rating": 45, "chemistry": 40},
        {"position": "LW", "overall_rating": 68, "pace": 78, "dribbling": 70, "attack_rating": 66, "defense_rating": 35, "chemistry": 35},
        {"position": "RW", "overall_rating": 67, "pace": 76, "dribbling": 68, "attack_rating": 64, "defense_rating": 33, "chemistry": 35},
        {"position": "ST", "overall_rating": 70, "pace": 80, "dribbling": 72, "attack_rating": 70, "defense_rating": 32, "chemistry": 40},
    ]

    features_strong = build_team_features(strong_players, "4-3-3")
    features_weak = build_team_features(weak_players, "4-3-3")

    result = predict(features_strong, features_weak, clf)
    print(f"Strong vs Weak prediction:")
    print(f"  Win: {result['win_prob']:.2%}  Draw: {result['draw_prob']:.2%}  Loss: {result['loss_prob']:.2%}")
    print(f"  Predicted scoreline: {result['predicted_scoreline']}")
    print(f"  Predicted outcome: {result['predicted_outcome']}")

    if result["win_prob"] > 0.5:
        print("  [PASS] Sanity check PASSED: strong team predicted to win")
    else:
        print("  [FAIL] Sanity check FAILED: strong team NOT predicted to win -- model may need tuning")


if __name__ == "__main__":
    main()
