# features/tfidf_features.py
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_tfidf_features(df, max_features=5000, ngram_range=(1,2)):
    vectorizer = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        lowercase=True,
        stop_words='english'
    )
    tfidf_matrix = vectorizer.fit_transform(df["essay_text"])
    return tfidf_matrix, vectorizer
