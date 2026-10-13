"""Configuration constants for SquadPredict."""

from pathlib import Path
from dotenv import load_dotenv

# ── Paths ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
SAMPLE_SCREENSHOTS_DIR = PROJECT_ROOT / "sample_screenshots"

DB_PATH = DATA_DIR / "squadpredict.db"
PLAYER_REFERENCE_CSV = DATA_DIR / "player_reference.csv"
SYNTHETIC_TRAINING_CSV = DATA_DIR / "synthetic_training_data.csv"
CLASSIFIER_PATH = MODELS_DIR / "classifier.pkl"

# ── OCR Crop Regions (x1, y1, x2, y2 as fractions of image size) ──────
# These are initial defaults — calibrate against real screenshots.
FORMATION_REGION = (0.0, 0.0, 1.0, 0.15)
PLAYER_LIST_REGION = (0.0, 0.15, 1.0, 0.95)

# ── OCR Settings ──────────────────────────────────────────────────────
MIN_UPSCALE_WIDTH = 1000  # Upscale 2x if image width is below this
OCR_CONFIDENCE_THRESHOLD = 0.6  # Below this, flag for manual review
PLAYER_MATCH_THRESHOLD = 80  # rapidfuzz score threshold

# ── Known Formations Whitelist ────────────────────────────────────────
KNOWN_FORMATIONS: list[str] = [
    "4-3-3",
    "4-2-3-1",
    "3-4-3",
    "4-4-2",
    "4-2-1-3",
    "5-3-2",
    "5-2-1-2",
    "4-1-4-1",
    "3-2-3-2",
    "4-3-1-2",
    "4-2-2-2",
    "4-1-2-3",
    "3-3-4",
    "4-4-1-1",
    "4-1-3-2",
    "5-4-1",
    "3-5-2",
    "4-5-1",
    "5-2-3",
]

# ── Formation Attack Bias ─────────────────────────────────────────────
# 0.0 = fully defensive, 1.0 = fully attacking
FORMATION_ATTACK_BIAS: dict[str, float] = {
    "3-3-4": 0.85,
    "3-4-3": 0.80,
    "4-2-1-3": 0.75,
    "4-1-2-3": 0.75,
    "4-3-3": 0.70,
    "4-2-3-1": 0.65,
    "4-4-2": 0.55,
    "4-3-1-2": 0.60,
    "3-2-3-2": 0.55,
    "4-4-1-1": 0.50,
    "4-2-2-2": 0.55,
    "4-1-4-1": 0.45,
    "4-5-1": 0.40,
    "3-5-2": 0.50,
    "5-2-1-2": 0.40,
    "4-1-3-2": 0.55,
    "5-3-2": 0.35,
    "5-2-3": 0.50,
    "5-4-1": 0.30,
}

# ── Simulator Settings ────────────────────────────────────────────────
BASE_GOAL_RATE = 1.3  # Average goals per team per eFootball match
DEFAULT_N_SIMS = 1000  # Number of Monte Carlo simulations
UPSET_FACTOR_STD = 0.15  # Std dev of random upset noise
CHEMISTRY_BONUS_SCALE = 0.05  # Chemistry bonus multiplier

# ── Model Settings ────────────────────────────────────────────────────
REAL_DATA_WEIGHT = 3.0  # Weight multiplier for real match data in retraining
SYNTHETIC_DATA_COUNT = 10_000  # Number of synthetic training pairs to generate
