import re
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

STOP = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

def clean_text(s: str) -> str:
    if pd.isna(s):
        return ""
    s = re.sub(r"http\S+|www\S+", "", s)
    s = re.sub(r"[^a-zA-Z\s]", "", s)
    s = s.lower().strip()
    tokens = [w for w in s.split() if w not in STOP]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

def load_and_process(path: str, text_col: str = 'review', label_col: str = 'label') -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df[[text_col, label_col]].dropna()
    df['clean'] = df[text_col].apply(clean_text)
    return df
