import os
import logging
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix, f1_score

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_baseline():
    data_path = "ml/data/cleaned_tickets.csv"
    if not os.path.exists(data_path):
        logging.error("Cleaned data not found. Run ml/preprocessing.py ")
        return
    df = pd.read_csv(data_path)
    X = df['text']
    y = df['category']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    logging.info(f"Train size: {len(X_train)} | Test size: {len(X_test)}")
    
    logging.info("Building TF-IDF + Logistic Regression pipeline")
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2), stop_words='english')),
        ('clf', LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42))
    ])
    logging.info("Training model")
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    logging.info("="*50)
    logging.info(f"Macro-F1 Score: {macro_f1:.4f}")
    logging.info("="*50)
    print("\nPer-Class Performance:")
    print(classification_report(y_test, y_pred))
    
    model_dir = "ml/models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "baseline_classifier.pkl")
    joblib.dump(pipeline, model_path)
    logging.info(f"Baseline model saved to {model_path}")

if __name__ == "__main__":
    train_baseline()