from fastapi import FastAPI, Request, status
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from pathlib import Path
import os
import pickle
from schemas import FertilizerModel,RainFallModel,RainFallPrediction,FertilizerPrediction,CropModel,CropPrediction


BASE_DIR = Path(__file__).resolve().parent
print("BASE DIR IS ********-", BASE_DIR)

STATICFILES_DIR = os.path.join(BASE_DIR, "static")
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")



app = FastAPI()

# STATIC FILES
app.mount("/static", StaticFiles(directory=STATICFILES_DIR), name="static")

# TEMPLATES
templates = Jinja2Templates(directory=TEMPLATES_DIR)


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.get("/services", response_class=HTMLResponse)
async def services(request: Request):
    return templates.TemplateResponse("services.html", {"request": request})


@app.post("/services/crop-prediction", response_model=CropPrediction, status_code=status.HTTP_200_OK)
async def predict_crop(crop_model: CropModel):
    """
    Crop Prediction
    """

    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "crop_recommender.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        crop = model.predict([[crop_model.N, crop_model.P, crop_model.K, crop_model.temperature,
                             crop_model.humidity, crop_model.ph, crop_model.rainfall]])
        return CropPrediction(crop=crop[0])
    except:
        status_code = status.HTTP_404_NOT_FOUND
        return CropPrediction(crop="Not Found")


@app.post("/services/fertilizer-prediction", response_model=FertilizerPrediction, status_code=status.HTTP_200_OK)
async def predict_fertilizer(fertilizer_model: FertilizerModel):
    """
    Fertilizer Prediction
    """
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models",
                              "fertilizer_prediction.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        fertilizer = model.predict([[fertilizer_model.Temparature, fertilizer_model.Humidity, fertilizer_model.Moisture, fertilizer_model.Soil_Type,
                                   fertilizer_model.Crop_Type, fertilizer_model.Nitrogen, fertilizer_model.Potassium, fertilizer_model.Phosphorous]])
        return FertilizerPrediction(Fertilizer_Name=fertilizer[0])
    except:
        status_code = status.HTTP_400_BAD_REQUEST
        return FertilizerPrediction(Fertilizer_Name="Not Found")


@app.post("/services/rainfall-prediction", response_model=RainFallPrediction, status_code=status.HTTP_200_OK)
async def predict_rainfall(rainfall_model: RainFallModel):
    """
    RainFall Prediction
    """
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "rain_prediction.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        rainfall = model.predict([[rainfall_model.avgTemp, rainfall_model.maxTemp, rainfall_model.minTemp, rainfall_model.precipitation, rainfall_model.avgWindSpeed,
                                 rainfall_model.maxWindSpeed, rainfall_model.maxWindSpeedDir, rainfall_model.maxInstWindSpeed, rainfall_model.maxInstWindSpeedDir, rainfall_model.minAtmosPressure]])
        return RainFallPrediction(RainFall=rainfall[0])
    except:
        status_code = status.HTTP_400_BAD_REQUEST
        return RainFallPrediction(RainFall="Not Found")
