# importing required dependencies
import os 
from dotenv import load_dotenv

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pathlib import Path
import pickle
from pydantic import BaseModel
import numpy as np

load_dotenv()

frontend_url = os.getenv("FRONTEND_URL")

# defining the input data model for house price prediction
class HousePricePredictionInput(BaseModel):
    total_sqft: float
    bath: int
    balcony: int
    bhk: int
    area_type: str
    location: str
    
# initiating fastapi app
app = FastAPI(
    title="FastApi Template",
    description="A simple FastAPI template",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[frontend_url],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# getting the pickle file's path
MODEL_PATH = Path(__file__).resolve().parent / "models" / "bengaluru_house_price_linear_regression_model.pickle"

# loading the pickle file and extracting the model and feature columns
try:
    with open(MODEL_PATH, "rb") as file:
        model_package = pickle.load(file)
        
        model = model_package["model"]
        feature_columns = model_package["feature_columns"]
        
except FileNotFoundError:
    raise RuntimeError(f"The pickle file was not found at the specified path: {MODEL_PATH}")

except KeyError:
    raise RuntimeError("The pickle file does not contains the expected keys: 'model' and 'feature_columns'")

except Exception as e:
    raise RuntimeError(f"An unexpected error occurred: {e}")

# classifying the feature columns into numerical, area and location types
NUMERICAL_TYPE_FEATURES = ["total_sqft", "bath", "balcony", "bhk"]

AREA_TYPE_FEATURES = ["built-up  Area", "Plot  Area", "Super built-up  Area", "Carpet  Area"]

# list comprehension
LOCATION_TYPE_FEATURES = [
    feature for feature in feature_columns if feature not in NUMERICAL_TYPE_FEATURES and feature not in AREA_TYPE_FEATURES
]

# forming the input row for the trained model
def create_input_row(data: HousePricePredictionInput):
    input_row = np.zeros(len(feature_columns))
    
    # for numerical features
    numerical_features = {
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk
    }
    
    for feature, value in numerical_features.items():
        try: 
            feature_index = feature_columns.index(feature)
            input_row[feature_index] = value
        except ValueError:
            pass
               
    
    # for area type features    
    cleaned_area_type = data.area_type.strip().lower()
    matched_area = next((col for col in feature_columns if col.strip().lower() == cleaned_area_type), None)
    
    if matched_area:
        area_type_index = feature_columns.index(matched_area)
        input_row[area_type_index] = 1
    else:
        raise HTTPException(status_code=400, detail=f"Invalid area type received: '{data.area_type}'")
         

    # for location type features
    cleaned_location = data.location.strip().lower()
    matched_location = next((col for col in feature_columns if col.strip().lower() == cleaned_location), None)

    if matched_location:
        location_type_index = feature_columns.index(matched_location)
        input_row[location_type_index] = 1
    else:
        raise HTTPException(status_code=400, detail=f"Invalid location received: '{data.location}'")

    return input_row


# endpoints
@app.get("/")
def read_root():
    return {"massage":"This is the root of house price prediction API"}

@app.get("/health")
def read_health():
    return {"status": "ok"}

@app.get("/model-info")
def read_model_info():
    return {
        "model_type": type(model).__name__,
        "no_of_feature_columns": len(feature_columns),
        "sample_feature_columns": feature_columns[:10]
    }

@app.get("/options") 
def read_options():
    return {
        "area_type_features": AREA_TYPE_FEATURES,
        "location_type_features": LOCATION_TYPE_FEATURES
    }
 
@app.post("/predict") 
def predict(data: HousePricePredictionInput):
    print("Received Data", data)
    
    input_row = create_input_row(data)
    prediction_result = model.predict([input_row])[0]
    
    return{
        "predicted_price": round(prediction_result, 2),
           "input_data": {
               "total_sqft": data.total_sqft,
               "bath": data.bath,
               "balcony": data.balcony,
               "bhk": data.bhk,
               "area_type": data.area_type,
               "location": data.location
           }
    }
           
   
          