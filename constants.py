# config/constants.py

# === File & Folder Paths ===
DATA_CLEANED_PATH = "data/essays_cleaned.csv"
FEATURES_DIR = "extract_features"
DATA_FOLD_PATH = "data/essays_folds.csv"

# === Feature Flags ===
INCLUDE_HANDCRAFTED = True
INCLUDE_TFIDF = True
INCLUDE_EMBEDDING = True

# === Default Model Settings ===
DEFAULT_FOLDS = 5
DEFAULT_MODEL_TYPE = "RandomForest"
MODE_ClASSICAL = "classical"
MODE_EMBEDDING = "embedding"
MODE_HYBRID = "hybrid"

# === Column Names ===
TEXT_COLUMN = "essay_text"
SCORE_COLUMN = "score"
CLASS_COLUMN = "class"

# === Hybrid Model Settings ===
HYBRID_FEATURES = ["hc", "embed"]

# === Other ===
RANDOM_SEED = 42
