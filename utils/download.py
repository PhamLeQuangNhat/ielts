from datasets import load_dataset
import pandas as pd
import os

dataset = load_dataset("chillies/IELTS-writing-task-2-evaluation", split="train")
df = dataset.to_pandas()

keep_cols = ['prompt', 'essay', 'evaluation', 'band']

os.makedirs("data", exist_ok=True)
df.to_csv("../data/essays.csv", index=False, encoding="utf-8")
print(f"Saved {len(df)} essays to data/essays.csv with columns: {list(df.columns)}")