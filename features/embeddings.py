from sentence_transformers import SentenceTransformer

def extract_sentence_embeddings(df, model_name="all-MiniLM-L6-v2"):
    model = SentenceTransformer(model_name)
    embeddings = model.encode(df["essay_text"].tolist(), batch_size=32, show_progress_bar=True)
    return embeddings
