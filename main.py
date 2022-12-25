from fastapi import FastAPI, Request, status,Form
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

def cropPred(crop_model: CropModel):
    MODEL_PATH = os.path.join(BASE_DIR, "ml_models", "crop_recommender.pkl")
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)

    try:
        crop = model.predict([[crop_model.N, crop_model.P, crop_model.K, crop_model.temperature,
                             crop_model.humidity, crop_model.ph, crop_model.rainfall]])
        return CropPrediction(crop=crop[0])
    except:
        return CropPrediction(crop="Not Found")

@app.get('/crop-prediction', response_class=HTMLResponse)
async def cropPredForm(request: Request):
    return templates.TemplateResponse("cropPred.html", {"request": request})

@app.post("/crop-prediction", response_model=CropPrediction, status_code=status.HTTP_200_OK)
async def cropPredForm(request: Request,nitrogen: float, phosphorus: float, potassium: float, temperature: float, humidity: float, ph: float, rainfall: float):
    try:
        
        cropPredResult = CropModel(N=nitrogen, P=phosphorus, K=potassium, temperature=temperature, humidity=humidity, ph=ph, rainfall=rainfall)
        predictedCrop = cropPred(cropPredResult)
        print(predictedCrop)

        return templates.TemplateResponse("cropPred.html", {"request": request, "cropPredResult": cropPredResult})
    
    except ValueError:
        return HTMLResponse(status_code=400, content="Invalid form field value")




@app.get('/register', response_class=HTMLResponse)
async def register(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})


@app.get('/login', response_class=HTMLResponse)
async def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})



