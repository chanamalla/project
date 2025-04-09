import pandas as pd
import numpy as np
import tensorflow as tf
import joblib

# Load trained ANN model & scaler
MODEL = tf.keras.models.load_model("model/ann_model.h5")
SCALER = joblib.load("model/scaler.pkl")

def predict(rainfall, pesticide, temperature):
    """
    Predicts crop yield using a trained ANN model.

    Args:
        rainfall (float): Average annual rainfall (mm)
        pesticide (float): Pesticide usage (tons)
        temperature (float): Average temperature (°C)

    Returns:
        float: Predicted yield
    """

    # Create a DataFrame for prediction
    predictdf = pd.DataFrame({
        "Avg_rainfall_mm_per_year": [rainfall],
        "Pesticide_Tons": [pesticide],
        "Avg_temp": [temperature]
    })

    # Standardize input data
    input_scaled = SCALER.transform(predictdf)

    # Predict yield
    prediction = MODEL.predict(input_scaled)[0][0]  # Extract scalar value

    return round(prediction, 2)  # Return rounded yield value
