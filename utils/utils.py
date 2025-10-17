from sklearn.metrics import classification_report, accuracy_score, mean_absolute_error, cohen_kappa_score
import numpy as np
import os
import pandas as pd
import numpy as np
from scipy.sparse import csr_matrix, hstack, load_npz, save_npz

from features.handcrafted import extract_handcrafted_features
from features.tfidf import extract_tfidf_features
from features.embeddings import extract_sentence_embeddings
from models.classical import ClassicalClassifier
from models.hybrid_model import HybridStackingRegressor
from utils.fold import get_fold_indices
from constants import *

def evaluate(y_test, y_pred):
    mae = mean_absolute_error(y_test, y_pred)
    y_pred_band = np.clip(np.round(y_pred*2)/2, 4.0, 9.0)
    y_test_band = y_test.astype(float) 

    y_pred_int = ((y_pred_band - 4.0) * 2).astype(int)
    y_test_int = ((y_test_band - 4.0) * 2).astype(int)

    acc = accuracy_score(y_test_int, y_pred_int)
    qwk = cohen_kappa_score(y_test_int, y_pred_int, weights="quadratic")

    print(f"Accuracy: {acc:.3f}, MAE: {mae:.3f}, QWK: {qwk:.3f}")
    print(classification_report(y_test_int, y_pred_int, digits=3))

    return {"accuracy": acc, "mae": mae, "qwk": qwk}

def get_features(df: pd.DataFrame, save_dir=FEATURES_DIR, include_handcrafted=True, include_tfidf=True, include_embed=True):
    os.makedirs(save_dir, exist_ok=True)
    features = {}

    if include_handcrafted:
        path = os.path.join(save_dir, "handcrafted.npz")
        if os.path.exists(path):
            features['hc'] = load_npz(path)
            print("Loaded handcrafted features")
        else:
            features['hc'] = csr_matrix(extract_handcrafted_features(df).fillna(0))
            save_npz(path, features['hc'])
            print("Extracted & saved handcrafted features")

    if include_tfidf:
        path = os.path.join(save_dir, "tfidf.npz")
        if os.path.exists(path):
            features['tfidf'] = load_npz(path)
            print("Loaded TF-IDF features")
        else:
            X_tfidf, _ = extract_tfidf_features(df)
            features['tfidf'] = X_tfidf
            save_npz(path, X_tfidf)
            print("Extracted & saved TF-IDF features")

    if include_embed:
        path = os.path.join(save_dir, "embedding.npz")
        if os.path.exists(path):
            features['embed'] = load_npz(path)
            print("Loaded embedding features")
        else:
            features['embed'] = csr_matrix(extract_sentence_embeddings(df))
            save_npz(path, features['embed'])
            print("Extracted & saved embedding features")

    return features

def run_pipeline(df, folds, X_dict, y, model_type, model_class):
    df, fold_indices = get_fold_indices(df, folds)
    results = {}

    for fold_num, (train_idx, test_idx) in enumerate(fold_indices):
        y_train, y_test = y[train_idx], y[test_idx]

        if issubclass(model_class, HybridStackingRegressor):
            X_train = (X_dict['hc'][train_idx].toarray(), X_dict['embed'][train_idx].toarray())
            X_test = (X_dict['hc'][test_idx].toarray(), X_dict['embed'][test_idx].toarray())
            model = model_class()
            model.fit(*X_train, y_train)
            y_pred = model.predict(*X_test)
        else:
            features_to_use = [X_dict[k] for k in ['hc','tfidf','embed'] if k in X_dict]
            X_combined = hstack(features_to_use)
            X_train_combined = X_combined[train_idx]
            X_test_combined = X_combined[test_idx]
            model = model_class(model_type=model_type) if issubclass(model_class, ClassicalClassifier) else model_class()
            model.fit(X_train_combined, y_train)
            y_pred = model.predict(X_test_combined)

        fold_result = evaluate(y_test, y_pred)
        results[f"Fold{fold_num+1}"] = fold_result

    print(f"\n{model_type} Summary:")
    print(pd.DataFrame(results).T)
    return results
