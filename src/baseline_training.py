from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import joblib

import data

def train_and_save_baseline():
    X_train, X_test, y_train, y_test = data.get_train_test_splits()
    model = Pipeline([
        ('tfidf', TfidfVectorizer()),
        ('logreg', LogisticRegression())
    ])

    model.fit(X_train, y_train)
    joblib.dump(model, '../models/tfidf_logreg_baseline.joblib')
    print('Baseline model saved')



if __name__ == "__main__":
    train_and_save_baseline()