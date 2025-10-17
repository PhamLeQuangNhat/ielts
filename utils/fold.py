import pandas as pd
from sklearn.model_selection import StratifiedKFold
import os
from constants import DATA_FOLD_PATH
def create_and_save_folds(df, target_col="score", n_splits=5, save_path="data/essays_folds.csv"):
    df = df.copy()
    if "class" not in df.columns:
        df["class"] = df[target_col].apply(lambda x: int((x-4.0)/0.5))
        print(df["class"].value_counts().sort_index())
        input()

    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    df["fold"] = -1

    for fold_number, (_, val_idx) in enumerate(skf.split(df, df["class"])):
        df.loc[val_idx, "fold"] = fold_number

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"Fold assignment saved to {save_path}")
    return df

def load_folds(csv_path=DATA_FOLD_PATH):
    df = pd.read_csv(csv_path)
    if "fold" not in df.columns:
        raise ValueError("CSV không chứa cột 'fold'")
    return df

def get_fold_indices(df: pd.DataFrame, n_splits: int):
    fold_csv_path = DATA_FOLD_PATH
    if os.path.exists(fold_csv_path):
        print("Loading existing folds...")
        df = load_folds(fold_csv_path)
    else:
        print("Creating folds and saving...")
        df = create_and_save_folds(df, target_col="score", n_splits=n_splits, save_path=fold_csv_path)

    folds = []
    for fold_number in range(df["fold"].nunique()):
        train_idx = df.index[df["fold"] != fold_number].to_numpy()
        test_idx = df.index[df["fold"] == fold_number].to_numpy()
        folds.append((train_idx, test_idx))

    return df, folds
