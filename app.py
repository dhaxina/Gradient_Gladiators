import joblib
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

# Load Model
try:
    model = joblib.load("./model/spam_model.0.1.0.pkl")
    print("Model Loaded Successfully!")
except Exception as e:
    print(f"Error Loading Model: {e}")

# Input Schema
class EmailInput(BaseModel):
    text: str

@app.get("/")
def home():
    return {"message": "Spam Classification API is Running!"}

@app.post("/predict")
def predict_spam(email: EmailInput):
    prediction = model.predict([email.text])[0]
    return {"spam": bool(prediction)}

if __name__ == "__main__":
    print("Running FastAPI server")
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
