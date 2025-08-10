from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

def train_model(X_train, y_train, pipeline, model_path='models/lr_tfidf.pkl'):
    clf = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    full_pipeline = Pipeline([('features', pipeline), ('clf', clf)])
    full_pipeline.fit(X_train, y_train)
    joblib.dump(full_pipeline, model_path)
    return full_pipeline

def load_model(path='models/lr_tfidf.pkl'):
    return joblib.load(path)
