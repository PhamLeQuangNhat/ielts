import pandas as pd
import re
import os
from nltk.tokenize import word_tokenize, sent_tokenize
import nltk

nltk.download("punkt", quiet=True)

def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "essay_text" not in df.columns and "essay" in df.columns:
        df["essay_text"] = df["essay"].astype(str)
    return df

def normalize_band(x):
    if pd.isna(x): return None
    x = str(x).strip()
    if x.startswith("<"): return None
    x = re.sub(r"[^\d\.]", "", x)
    if x == "": return None
    try:
        val = float(x)
        return val if val >= 4 else None
    except:
        return None

def clean_text(text: str) -> str:
    text = str(text)
    text = text.replace("\n", " ").replace("\r", " ").strip()
    text = re.sub(r"\s+", " ", text)
    return text.lower()

def clean_dataset(df: pd.DataFrame, min_words=250) -> pd.DataFrame:
    if "band" in df.columns:
        df["score"] = df["band"].apply(normalize_band)
    df = df.dropna(subset=["essay_text"])
    df = df[df["essay_text"].str.strip() != ""]
    df["essay_text"] = df["essay_text"].apply(clean_text)
    df["num_words"] = df["essay_text"].apply(lambda x: len(word_tokenize(x)))
    df["num_sentences"] = df["essay_text"].apply(lambda x: len(sent_tokenize(x)))

    df = df[(df["num_words"] >= min_words)]

    if "score" in df.columns:
        df = df.dropna(subset=["score"])

    return df

def main():
    input_path = "../data/essays.csv"
    output_path = "../data/essays_cleaned.csv"

    if not os.path.exists(input_path):
        raise FileNotFoundError(f"Không tìm thấy file: {input_path}")

    df = load_dataset(input_path)
    print(f"Tổng số bài gốc: {len(df)}")

    df = clean_dataset(df)
    print(f"Sau khi làm sạch: {len(df)} bài")
    print(df["score"].describe())
    print(df["num_words"].describe())

    columns_to_keep = ["prompt", "evaluation", "essay_text", "score", "num_words", "num_sentences"]
    df = df[columns_to_keep]

    df.to_csv(output_path, index=False, encoding="utf-8")
    print(f"File cleaned đã lưu tại: {output_path}")

if __name__ == "__main__":
    main()
