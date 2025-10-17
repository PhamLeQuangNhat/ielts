# main.py
import argparse

from utils.preprocessing import load_dataset
from models.embedding_regressor import EmbeddingRegressor
from models.classical import ClassicalClassifier
from models.hybrid_model import HybridStackingRegressor
from utils.utils import run_pipeline, get_features
from utils.fold import get_fold_indices
from constants import *

def run_classical_pipeline(df, folds=5, model_type=DEFAULT_MODEL_TYPE):
    print("⚡ Running classical ML classification...")
    df["class"] = df["score"].apply(lambda x: int((x - 4.0)/0.5))
    df = df.dropna(subset=["class"])
    y_class = df["class"].values
    X_dict = get_features(df, include_handcrafted=True, include_tfidf=True, include_embed=True)
    run_pipeline(df, folds, X_dict, y_class, model_type, ClassicalClassifier)

def run_embedding_pipeline(df, folds=5, model_type="Ridge"):
    print("Running embedding-only regression...")
    df, _ = get_fold_indices(df, folds)
    X_dict = get_features(df, include_handcrafted=False, include_tfidf=False, include_embed=True)
    y_score = df["score"].values
    run_pipeline(df, folds, X_dict, y_score, model_type, EmbeddingRegressor)

def run_hybrid_pipeline(df, folds=5):
    print("Running hybrid pipeline (handcrafted + embeddings)...")
    df, _ = get_fold_indices(df, folds)
    X_dict = get_features(df, include_handcrafted=True, include_tfidf=False, include_embed=True)
    y_score = df["score"].values
    run_pipeline(df, folds, X_dict, y_score, "HybridStacking", HybridStackingRegressor)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default=MODE_ClASSICAL, choices=["classical","embedding","hybrid"])
    parser.add_argument("--folds", type=int, default=DEFAULT_FOLDS)
    parser.add_argument("--model_type", type=str, default=DEFAULT_MODEL_TYPE,
                    help="Model type for classical or embedding pipeline")
    args = parser.parse_args()

    df = load_dataset(DATA_CLEANED_PATH)
    df = df.dropna(subset=["essay_text","score"])

    if args.mode == MODE_ClASSICAL:
        run_classical_pipeline(df, folds=args.folds, model_type=args.model_type)
    elif args.mode == MODE_EMBEDDING:
        run_embedding_pipeline(df, folds=args.folds, model_type=args.model_type)
    elif args.mode == MODE_HYBRID:
        run_hybrid_pipeline(df, folds=args.folds)

if __name__ == "__main__":
    main()
