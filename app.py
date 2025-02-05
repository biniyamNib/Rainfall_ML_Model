from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import pandas as pd
import joblib
import asyncio
from sklearn.pipeline import Pipeline

# Load the trained model
model = joblib.load("rain_prediction_model.joblib")  # Replace with the path to your saved model

# Define the input data schema using Pydantic
class InputData(BaseModel):
    MinTemp: float
    MaxTemp: float
    Rainfall: float
    Evaporation: float
    Sunshine: float
    WindGustDir: str
    WindGustSpeed: float
    WindDir9am: str
    WindDir3pm: str
    WindSpeed9am: float
    WindSpeed3pm: float
    Humidity9am: float
    Humidity3pm: float
    Pressure9am: float
    Pressure3pm: float
    Cloud9am: float
    Cloud3pm: float
    Temp9am: float
    Temp3pm: float
    RainToday: str

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://127.0.0.1:5500"],  # Allows all origins (replace "*" with specific origins in production)
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods
    allow_headers=["*"],  # Allows all headers
)

# Serve static files
app.mount("/static", StaticFiles(directory="static"), name="static")

# Simulate an asynchronous data fetch operation
async def fetch_data():
    await asyncio.sleep(1)  # Simulate an I/O operation
    return {"message": "Data fetched successfully"}

@app.options("/predict")
async def preflight():
    return {}

# Root endpoint
@app.get("/")
async def read_root():
    result = await fetch_data()  # Await the asynchronous function
    return result

# Define the prediction endpoint
@app.post("/predict")
def predict(data: InputData):
    try:
        # Convert input data to a DataFrame
        input_data = pd.DataFrame([data.dict()])

        # Make a prediction using the trained model
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        # Return the prediction
        return {
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability": float(prediction_proba[0])
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Save the trained model to a file (run this once before deploying)
# joblib.dump(best_model, "rainfall_model.pkl")
