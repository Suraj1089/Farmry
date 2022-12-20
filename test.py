import pickle
import os 
import pathlib
import warnings
warnings.filterwarnings("ignore")
import pandas as pd


BASE_DIR = pathlib.Path(__file__).resolve().parent


MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "crop_recommender.pkl")


def recommend_crop(N, P, K, temperature, humidity, ph, rainfall):
    """
    Recommend Crop
    """
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[N, P, K, temperature, humidity, ph, rainfall]])
    return prediction[0]

print("recommende crop->",recommend_crop(20, 10, 30, 50, 3330, 50, 120))

def rainfall_prediction(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    """
    RainFall Prediction
    """
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "rain_prediction.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    return prediction[0]

# print("rainfall prediction->",rainfall_prediction(20, 10, 30, 50, 3330, 50, 120, 20))

def get_dummies(df, col):
    """
    Get Dummies
    """
    df = pd.get_dummies(df, columns=[col], prefix=[col])
    return df

def fertilizer_prediction(Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous):
    """
    Fertilizer Prediction
    """
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "fertilizer_prediction.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    prediction = model.predict([[Temparature, Humidity, Moisture, Soil_Type, Crop_Type, Nitrogen, Potassium, Phosphorous]])
    return prediction[0]

print("fertilizer prediction->",fertilizer_prediction(20, 10, 30, 50, 3330, 50, 120, 20))