
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pickle
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, GridSearchCV
import os

app = FastAPI()

# Load the model and vectorizer
model_path = "model/spam_model.0.1.0.pkl"
vectorizer_path = "model/vectorizer.0.1.0.pkl"

with open(model_path, "rb") as f:
    model = pickle.load(f)

with open(vectorizer_path, "rb") as f:
    vectorizer = pickle.load(f)

# Input structure for prediction
class InputData(BaseModel):
    sms: str

# Load dataset
df = pd.read_csv("dataset/SpamDetection.csv", encoding="latin-1")

# Endpoint for getting the best model parameters
@app.get("/best_model_parameters")
def get_best_model_parameters():
    try:
        return {"model": model.__class__.__name__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for predictions
@app.post("/predict")
def predict(input_data: InputData):
    try:
        sms = input_data.sms
        sms_vec = vectorizer.transform([sms])
        prediction = model.predict(sms_vec)
        return {"prediction": prediction[0]}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Endpoint for training
@app.post("/train")
def train():
    try:
        # Preprocess data
        X_train, X_test, y_train, y_test = train_test_split(df["sms"], df["label"], test_size=0.2, random_state=42)
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
                "params": {"C": [0.1, 1, 10], "max_iter": [1000, 2000]}
            },
            "DecisionTree": {
                "model": DecisionTreeClassifier(),
                "params": {"max_depth": [None, 10, 20], "min_samples_split": [2, 5, 10]}
            }
        }

        # MLflow tracking
        mlflow.set_experiment("Spam or Not Spam")
        best_accuracy = 0
        best_model = None

        for model_name, model_info in models.items():
            with mlflow.start_run(run_name=model_name):
                clf = GridSearchCV(model_info["model"], model_info["params"], cv=3, scoring="accuracy")
                clf.fit(X_train_vec, y_train)

                # Evaluate model
                y_pred = clf.predict(X_test_vec)
                accuracy = accuracy_score(y_test, y_pred)

                mlflow.log_params(clf.best_params_)
                mlflow.log_metrics({"accuracy": accuracy})
                mlflow.sklearn.log_model(clf.best_estimator_, model_name)

                if accuracy > best_accuracy:
                    best_accuracy = accuracy
                    best_model = clf.best_estimator_

        # Save the best model and vectorizer
        with open("model/spam_model.0.1.0.pkl", "wb") as f:
            pickle.dump(best_model, f)

        return {"message": "Training complete", "best_model": best_model.__class__.__name__}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
