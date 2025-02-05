from fastapi import FastAPI, HTTPException, Form, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
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

# Jinja2 templates setup
templates = Jinja2Templates(directory="templates")

# Simulate an asynchronous data fetch operation
async def fetch_data():
    await asyncio.sleep(1)  # Simulate an I/O operation
    return {"message": "Data fetched successfully"}

@app.options("/predict")
async def preflight():
    return {}

# Root endpoint
@app.get("/")
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Define the prediction endpoint
@app.post("/predict")
async def predict(request: Request, MinTemp: float = Form(...), MaxTemp: float = Form(...), Rainfall: float = Form(...),
                  Evaporation: float = Form(...), Sunshine: float = Form(...), WindGustDir: str = Form(...),
                  WindGustSpeed: float = Form(...), WindDir9am: str = Form(...), WindDir3pm: str = Form(...),
                  WindSpeed9am: float = Form(...), WindSpeed3pm: float = Form(...), Humidity9am: float = Form(...),
                  Humidity3pm: float = Form(...), Pressure9am: float = Form(...), Pressure3pm: float = Form(...),
                  Cloud9am: float = Form(...), Cloud3pm: float = Form(...), Temp9am: float = Form(...),
                  Temp3pm: float = Form(...), RainToday: str = Form(...)):

    try:
        # Prepare input data
        input_data = pd.DataFrame([{
            "MinTemp": MinTemp,
            "MaxTemp": MaxTemp,
            "Rainfall": Rainfall,
            "Evaporation": Evaporation,
            "Sunshine": Sunshine,
            "WindGustDir": WindGustDir,
            "WindGustSpeed": WindGustSpeed,
            "WindDir9am": WindDir9am,
            "WindDir3pm": WindDir3pm,
            "WindSpeed9am": WindSpeed9am,
            "WindSpeed3pm": WindSpeed3pm,
            "Humidity9am": Humidity9am,
            "Humidity3pm": Humidity3pm,
            "Pressure9am": Pressure9am,
            "Pressure3pm": Pressure3pm,
            "Cloud9am": Cloud9am,
            "Cloud3pm": Cloud3pm,
            "Temp9am": Temp9am,
            "Temp3pm": Temp3pm,
            "RainToday": RainToday,
        }])

        # Make a prediction using the trained model
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)[:, 1]

        return JSONResponse(content={
            "prediction": "Yes" if prediction[0] == 1 else "No",
            "probability": round(float(prediction_proba[0]) * 100, 2)  # Return probability as percentage
        })
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

# Save the trained model to a file (run this once before deploying)
# joblib.dump(best_model, "rainfall_model.pkl")
