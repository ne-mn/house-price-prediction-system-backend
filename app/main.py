# importing required dependencies
from fastapi import FastAPI, HTTPException
from pathlib import Path
import pickle
from pydantic import BaseModel
import numpy as np

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
    input_row = np.array(len(feature_columns)).zeros()
    
    # for numerical features
    numerical_features = {
        "total_sqft": data.total_sqft,
        "bath": data.bath,
        "balcony": data.balcony,
        "bhk": data.bhk
    }
    
    for feature, value in numerical_features.items():
        feature_index = feature_columns.index(feature)
        input_row[feature_index] = value
    
    # for area type features    
    if data.area_type not in AREA_TYPE_FEATURES:
        return HTTPException(status_code=400, detail=f"Invalid area type: {data.area_type}. Valid options are: {AREA_TYPE_FEATURES}")
    
    area_type_index = feature_columns.index(data.area_type) 
    input_row[area_type_index] = 1          

    # for location type features
    if data.location not in LOCATION_TYPE_FEATURES:
            return HTTPException(status_code=400, detail=f"Invalid location: {data.location}. Valid options are: {LOCATION_TYPE_FEATURES}")

    location_type_index = feature_columns.index(data.location) 
    input_row[location_type_index] = 1

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
    
    input_row = create_input_row(data)
    prediction_result = model.predict([input_row])[0]
    
    return{
        "prediction": round(prediction_result, 2),
           "input_data": {
               "total_sqft": data.total_sqft,
               "bath": data.bath,
               "balcony": data.balcony,
               "bhk": data.bhk,
               "area_type": data.area_type,
               "location": data.location
           }
           }
           
   
          