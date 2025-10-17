# features/handcrafted.py
import pandas as pd
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk import pos_tag, ngrams
import nltk
import string
import re
from nltk.corpus import stopwords
from textstat import flesch_reading_ease, gunning_fog, smog_index, flesch_kincaid_grade
from spellchecker import SpellChecker

nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')
STOPWORDS = set(stopwords.words("english"))
AWL = set([
    "analyze", "approach", "area", "assess", "assume", "authority",
    "available", "benefit", "concept", "consistent", "constitutional"
])
CONNECTORS = set([
    "however", "moreover", "therefore", "thus", "furthermore",
    "consequently", "additionally", "meanwhile", "in contrast", "on the other hand"
])
SUB_CLAUSES = ["if", "because", "although", "while", "since", "when", "unless", "whereas"]

spell = SpellChecker()

def extract_handcrafted_features(df: pd.DataFrame) -> pd.DataFrame:
    features = pd.DataFrame()

    # --- A. Độ dài & cấu trúc cơ bản ---
    features["word_count"] = df["essay_text"].apply(lambda x: len(word_tokenize(x)))
    features["sentence_count"] = df["essay_text"].apply(lambda x: max(1,len(sent_tokenize(x))))
    features["avg_sentence_length"] = features["word_count"] / features["sentence_count"]
    features["max_sentence_length"] = df["essay_text"].apply(lambda x: max([len(word_tokenize(s)) for s in sent_tokenize(x)]))
    features["min_sentence_length"] = df["essay_text"].apply(lambda x: min([len(word_tokenize(s)) for s in sent_tokenize(x)]))
    features["paragraph_count"] = df["essay_text"].apply(lambda x: max(1, x.count("\n")+1))
    features["short_sentence_ratio"] = df["essay_text"].apply(lambda x: sum(1 for s in sent_tokenize(x) if len(word_tokenize(s)) < 8)/len(sent_tokenize(x)))
    features["type_token_ratio"] = df["essay_text"].apply(lambda x: len(set([w.lower() for w in word_tokenize(x) if w.isalpha()])) / max(len(word_tokenize(x)), 1))

    # --- B. Lexical Resource ---
    features["awl_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x.lower()) if w in AWL)/max(len(word_tokenize(x)),1))
    features["stopword_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x.lower()) if w in STOPWORDS)/max(len(word_tokenize(x)),1))

    def bigram_collocations_ratio(text):
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        bigrams = list(ngrams(words, 2))
        common_bigrams = set([("in","addition"),("as","well"),("due","to"),("according","to")])
        return sum(1 for bg in bigrams if bg in common_bigrams)/max(len(bigrams),1)
    features["collocation_ratio"] = df["essay_text"].apply(bigram_collocations_ratio)

    features["long_word_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x) if len(w)>=6)/max(len(word_tokenize(x)),1))
    features["avg_word_length"] = df["essay_text"].apply(lambda x: sum(len(w) for w in word_tokenize(x))/max(len(word_tokenize(x)),1))
    features["rare_word_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x.lower()) if w not in AWL and len(w)>3)/max(len(word_tokenize(x)),1))

    # --- C. Grammar & Complexity ---
    def pos_ratios(text):
        words = [w for w in word_tokenize(text) if w.isalpha()]
        tags = pos_tag(words)
        total = len(words)
        counts = {"noun_ratio":0,"verb_ratio":0,"adj_ratio":0,"adv_ratio":0}
        for _, tag in tags:
            if tag.startswith("NN"): counts["noun_ratio"] +=1
            elif tag.startswith("VB"): counts["verb_ratio"] +=1
            elif tag.startswith("JJ"): counts["adj_ratio"] +=1
            elif tag.startswith("RB"): counts["adv_ratio"] +=1
        for k in counts: counts[k] /= max(total,1)
        return pd.Series(counts)
    features = pd.concat([features, df["essay_text"].apply(pos_ratios)], axis=1)

    # Lỗi chính tả
    def spelling_error_ratio(text):
        words = [w for w in word_tokenize(text) if w.isalpha()]
        misspelled = spell.unknown(words)
        return len(misspelled)/max(len(words),1)
    features["spelling_error_ratio"] = df["essay_text"].apply(spelling_error_ratio)

    features["passive_ratio"] = df["essay_text"].apply(lambda x: len(re.findall(r"\b(is|are|was|were|be|been|being)\s+\w+ed\b", x.lower()))/max(len(sent_tokenize(x)),1))
    features["subordinate_clause_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x.lower()) if w in SUB_CLAUSES)/max(len(sent_tokenize(x)),1))
    features["pos_diversity"] = df["essay_text"].apply(lambda x: len(set(tag for _,tag in pos_tag(word_tokenize(x)))))

    # --- D. Coherence & Cohesion ---
    features["connector_ratio"] = df["essay_text"].apply(lambda x: sum(1 for w in word_tokenize(x.lower()) if w in CONNECTORS)/max(len(word_tokenize(x)),1))

    def sentence_overlap(text):
        sents = sent_tokenize(text)
        overlaps = []
        for i in range(len(sents)-1):
            set1 = set(word_tokenize(sents[i].lower()))
            set2 = set(word_tokenize(sents[i+1].lower()))
            overlaps.append(len(set1 & set2)/max(len(set1 | set2),1))
        return sum(overlaps)/max(len(overlaps),1)
    features["sentence_overlap"] = df["essay_text"].apply(sentence_overlap)

    def repeated_word_ratio(text):
        words = [w.lower() for w in word_tokenize(text) if w.isalpha()]
        counts = pd.Series(words).value_counts()
        return sum(1 for c in counts if c>1)/max(len(words),1)
    features["repeated_word_ratio"] = df["essay_text"].apply(repeated_word_ratio)

    # --- Readability ---
    features["flesch_reading_ease"] = df["essay_text"].apply(flesch_reading_ease)
    features["gunning_fog"] = df["essay_text"].apply(gunning_fog)
    features["smog_index"] = df["essay_text"].apply(smog_index)
    features["flesch_kincaid_grade"] = df["essay_text"].apply(flesch_kincaid_grade)

    # --- Style & Punctuation ---
    features["exclamation_ratio"] = df["essay_text"].apply(lambda x: x.count("!")/max(len(word_tokenize(x)),1))
    features["question_ratio"] = df["essay_text"].apply(lambda x: x.count("?")/max(len(word_tokenize(x)),1))
    features["comma_ratio"] = df["essay_text"].apply(lambda x: x.count(",")/max(len(word_tokenize(x)),1))
    return features
