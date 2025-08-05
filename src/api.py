from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import joblib
import numpy as np
from src.utils.logger import setup_logger
from src.utils.config import load_config

app = FastAPI()
logger = setup_logger()
config = load_config()

try:
    model = joblib.load('ridge_regression_model.pkl')  # Load best model
    EXPECTED_FEATURES = model.n_features_in_  # Ambil jumlah fitur yang diharapkan
    logger.info(f"Model loaded successfully. Expecting {EXPECTED_FEATURES} features.")
except FileNotFoundError:
    logger.error("Model file 'ridge_regression_model.pkl' not found. Please train the model first.")
    raise HTTPException(status_code=500, detail="Model not available. Please train the model first.")
except Exception as e:
    logger.error(f"Error loading model: {str(e)}")
    raise HTTPException(status_code=500, detail=f"Error loading model: {str(e)}")

class HouseInput(BaseModel):
    features: list[float]

@app.get("/")
async def root():
    """Root endpoint for testing server availability."""
    return {"message": "Welcome to House Price Prediction API", "status": "running", "expected_features": EXPECTED_FEATURES}

@app.post("/predict")
async def predict(house: HouseInput):
    try:
        logger.info("Received prediction request")
        if len(house.features) != EXPECTED_FEATURES:
            raise HTTPException(status_code=400, detail=f"X has {len(house.features)} features, but Ridge is expecting {EXPECTED_FEATURES} features as input.")
        input_data = pd.DataFrame([house.features])
        prediction_log = model.predict(input_data)[0]
        # Batasi prediksi log agar tidak menghasilkan nilai ekstrem
        if prediction_log > 20 or prediction_log < -20:  # Batas wajar untuk log(SalePrice)
            raise HTTPException(status_code=400, detail="Prediction out of reasonable range. Check input data.")
        prediction = np.expm1(prediction_log)
        logger.info(f"Prediction: {prediction}")
        return {"prediction": prediction}
    except HTTPException as e:
        logger.error(f"Validation error: {str(e.detail)}")
        raise e
    except Exception as e:
        logger.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))