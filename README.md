# Rainfall Predicting Model

## Project Overview

This project aims to predict whether it will rain tomorrow in using historical weather data. The problem is framed as a **binary classification task**, where the target variable is `RainTomorrow` (Yes/No). The project involves data exploration, preprocessing, model training, evaluation, and deployment.

## Table of Contents

 1. Installation
 2. Datset
 3. Usage
    <!-- - Exploratory Data Analysis (EDA)
    - Data Preprocessing
    - Model Traning
    - Model Evaluation
    - API Deployment -->
 4. Results
 5. Project Structure
 6. Future Improvements


 ## Installation

 #### Prerequisites

 - Python 3.8 or higher
 - pip (Python package installer)

 #### Steps

 1. Clone the repository:
   `git clone https://github.com/biniyamNib/Rainfall_Predicting_Model.git`
    `cd Rain_Prediction_Model`

 2. Install the required dependencies:
   `pip install -r requirements.txt`
  
## Dataset

  #### Souces
   - **Dataset**: [Rain in Australia](https://www.kaggle.com/jsphyg/weather-dataset-rattle-package)
   - **License**: [Open Database License (ODbL)](https://opendatacommons.org/licenses/odbl/)
  #### Download
  - Download the dataset from the link above and place it in the the folder

## Usage

 Execute the Python script to perform the entire workflow, including data loading, exploratory data analysis (EDA), preprocessing, model training, evaluation, and saving the model:

 `python main.py`

 #### Model Evaluation

 - The evaluation metrics (accuracy, precision, recall, F1-score, ROC-AUC) are displayed in the terminal after training.

 #### API Deployment

 - Deploy the model as an API using FastAPI:
 `uvicorn app:app --reload`
 - Access the API documentation at [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs).

## Results

#### Model Performance

| Metric	| Value |
| --------- | ----- |
| Accuracy	| 0.85  |
| Precision	| 0.78  |
| Recall	| 0.72  |
| F1-Score	| 0.75  |
| ROC-AUC	| 0.89  |

#### API Example

- Input:

```  
{
  "MinTemp": 10.0,
  "MaxTemp": 25.0,
  "Rainfall": 0.0,
  "Evaporation": 4.0,
  "Sunshine": 7.0,
  "WindGustDir": "NW",
  "WindGustSpeed": 30.0,
  "WindDir9am": "NW",
  "WindDir3pm": "NW",
  "WindSpeed9am": 10.0,
  "WindSpeed3pm": 15.0,
  "Humidity9am": 60.0,
  "Humidity3pm": 50.0,
  "Pressure9am": 1015.0,
  "Pressure3pm": 1013.0,
  "Cloud9am": 5.0,
  "Cloud3pm": 6.0,
  "Temp9am": 15.0,
  "Temp3pm": 20.0,
  "RainToday": "No"
}

```

- Output:
```
{
  "prediction": "No",
  "probability": 0.22
}

```

## Project Structure

```

Rain_Prediction_Model/
├── app.py                       # FastAPI application   
├── Document.pdf                 # Documentation
├── main.py                      # Script for data preprocessing and training the model
├── rain_prediction_model.joblib # Trained model
├── requirements.txt             # Python dependencies
├── Weather.csv                  # csv file
└── README.md  

```

## Future Improvements

1. **Handling Class Imbalance:**
 
   - Use techniques like SMOTE or class weighting to address imbalance.

2. **Advanced Models:**

   - Experiment with gradient  boosting models (e.g., XGBoost, LightGBM) or neural networks.

3. **Feature Engineering:**

   - Incorporate additional features like seasonality, geographic location, or time-based trends.

4. **Real-Time Data Integration:**

   - Integrate real-time weather data for more accurate predictions.

