"""Project-wide configuration for the DSCI 441 mask detection project."""
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
DEFAULT_MODEL_PATH = MODELS_DIR / "mask_model.joblib"
CACHE_DIR = PROJECT_ROOT / ".cache"

RANDOM_STATE = 441

# Higher than the first 64x64 version so HOG keeps more mask-edge detail.
# The default final model uses 128x128; use --image-size 96 96 if diagnostics suggest overfitting.
# This is still a classical, non-deep-learning image pipeline.
IMAGE_SIZE = (128, 128)

POSITIVE_LABEL = "WithMask"
NEGATIVE_LABEL = "WithoutMask"
CLASS_ORDER = [POSITIVE_LABEL, NEGATIVE_LABEL]
