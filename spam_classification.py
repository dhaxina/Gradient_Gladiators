import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn

import pickle

# Load dataset
df = pd.read_csv("dataset/SpamDetection.csv", encoding="latin-1")

# Preprocess data
X_train, X_test, y_train, y_test = train_test_split(df["sms"], df["label"], test_size=0.2, random_state=42)

# Vectorize text
vectorizer = TfidfVectorizer(stop_words="english")
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Define models and hyperparameters
models = {
    "MultinomialNB": {
        "model": MultinomialNB(),
        "params": {"alpha": [0.1, 0.5, 1.0, 2.0]}
    },
    "svm": {
        "model": LinearSVC(),
        "params": {"C": [0.1, 1, 10],"max_iter": [1000, 2000]}
    },
    "DecisionTree": {
        "model": DecisionTreeClassifier(),
        "params": {"max_depth": [None, 10, 20],"min_samples_split": [2, 5, 10]}
    }
}

# Train and evaluate models
mlflow.set_experiment("Spam or Not Spam")
best_accuracy = 0
best_model = None

for model_name, model_info in models.items():
    with mlflow.start_run(run_name=model_name):
        print(f"Training {model_name}...")
        clf = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring="accuracy")
        clf.fit(X_train_vec, y_train)

        # Evaluate model
        y_pred = clf.predict(X_test_vec)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        # Log metrics
        mlflow.log_params(clf.best_params_)
        mlflow.log_metrics({
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        })
        mlflow.sklearn.log_model(clf.best_estimator_, model_name)

        # Track the best model
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_model = clf.best_estimator_

# Save the best model and vectorizer using Pickle
with open("model/spam_model.0.1.0.pkl", "wb") as f:
    pickle.dump(best_model, f)
with open("model/vectorizer.0.1.0.pkl", "wb") as f:
    pickle.dump(vectorizer, f)

print(f"Best model: {best_model.__class__.__name__} with accuracy: {best_accuracy}")