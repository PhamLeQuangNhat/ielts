import pandas as pd
import matplotlib.pyplot as plt
import os
from nltk.tokenize import word_tokenize
import nltk
import re

nltk.download("punkt", quiet=True)

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "essay_text" not in df.columns and "essay" in df.columns:
        df["essay_text"] = df["essay"].astype(str)
    return df


def normalize_band(x):
    if pd.isna(x):
        return None
    if not isinstance(x, str):
        x = str(x)
    is_less_than = "<" in x
    x = re.sub(r"[^\d\.]", "", x)
    x = x.strip()
    if x == "":
        return None
    try:
        val = float(x)
        if is_less_than:
            return max(val - 0.5, 0.5)
        return val if 0 < val <= 9 else None
    except:
        return None


def clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    if "band" in df.columns:
        df["band"] = (
            df["band"]
            .astype(str)
            .str.replace(r"[\n\r\t\xa0\u200b]+", "", regex=True)
            .str.strip()
        )
        df["score"] = df["band"].apply(normalize_band)

    df = df.dropna(subset=["essay_text"])
    df = df[df["essay_text"].str.strip() != ""]
    df["essay_length"] = df["essay_text"].apply(lambda x: len(word_tokenize(str(x))))
    return df


def show_dataset_summary(df: pd.DataFrame):
    print("Dataset info:")
    print(f"- Tổng số bài viết: {len(df)}")
    print(f"- Cột có sẵn: {list(df.columns)}\n")
    if "score" in df.columns and df["score"].notna().sum() > 0:
        valid_df = df[df["score"].notna()]
        low_band = (valid_df["score"] < 4).sum()
        print("Phân bố điểm (Band Scores):")
        print(valid_df["score"].value_counts().sort_index())
        print(f"\n Trung bình: {valid_df['score'].mean():.2f}")
        print(f"Khoảng điểm: {valid_df['score'].min()} → {valid_df['score'].max()}")
        print(f"Số bài Band < 4: {low_band} ({low_band/len(valid_df)*100:.2f}%)\n")
    else:
        print("Không có dữ liệu điểm hợp lệ.\n")

    print("Thống kê độ dài bài viết (số từ):")
    print(df["essay_length"].describe().round(1))
    print()


def plot_band_distribution(df):
    os.makedirs("plots", exist_ok=True)
    valid_df = df[df["score"].notna()]
    band_counts = valid_df["score"].value_counts().sort_index()

    plt.figure(figsize=(8, 5))
    band_counts.plot(kind="bar", color="skyblue", edgecolor="black")
    plt.title("Band Score Distribution (Including <4)")
    plt.xlabel("Band Score")
    plt.ylabel("Number of Essays")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)

    path = os.path.join("plots", "band_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ phân bố band tại: {path}")


def plot_length_distribution(df):
    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8, 5))
    df["essay_length"].plot(kind="hist", bins=30, edgecolor="black", color="lightcoral")
    plt.title("Essay Length Distribution")
    plt.xlabel("Số từ trong bài viết")
    plt.ylabel("Số lượng bài")
    plt.grid(axis="y", alpha=0.3)

    path = os.path.join("plots", "essay_length_distribution.png")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()
    print(f"Đã lưu biểu đồ độ dài bài viết tại: {path}")


def main():
    os.makedirs("plots", exist_ok=True)

    csv_path = "data/essays.csv"
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Không tìm thấy file: {csv_path}")
    df = load_dataset(csv_path)

    df = clean_dataset(df)
    show_dataset_summary(df)

    if "score" in df.columns and df["score"].notna().sum() > 0:
        plot_band_distribution(df)

    plot_length_distribution(df)


if __name__ == "__main__":
    main()
