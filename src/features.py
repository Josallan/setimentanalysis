from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline

def build_pipeline(max_features=5000, ngram=(1,2)):
    tfidf = TfidfVectorizer(max_features=max_features, ngram_range=ngram)
    return Pipeline([('tfidf', tfidf)])
