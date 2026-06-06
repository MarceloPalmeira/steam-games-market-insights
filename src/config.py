import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))

DATASET_DIR = PROJECT_ROOT / "dataset"
CSV_PATH = DATASET_DIR / "games.csv"
JSON_PATH = DATASET_DIR / "games.json"

OUTPUT_DIR = PROJECT_ROOT / "outputs"
FIGURES_DIR = OUTPUT_DIR / "figures"
TABLES_DIR = OUTPUT_DIR / "tables"
MODELS_DIR = OUTPUT_DIR / "models"

ANALYSIS_DATE = "2026-06-06"
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 3

SUCCESS_TOP_SHARE = 0.30
SUCCESS_COMPONENTS = [
    "owners_midpoint",
    "total_reviews",
    "peak_ccu",
    "recommendations",
    "average_playtime_forever",
]

TARGET_COLUMN = "success_commercial"
SCORE_COLUMN = "commercial_success_score"

CORE_DEPENDENCIES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "matplotlib": "matplotlib",
    "seaborn": "seaborn",
    "scipy": "scipy",
    "joblib": "joblib",
}


def ensure_output_dirs() -> None:
    for path in [PROJECT_ROOT / ".mplconfig", OUTPUT_DIR, FIGURES_DIR, TABLES_DIR, MODELS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
