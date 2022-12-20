from datetime import datetime, timedelta
from typing import Union

from fastapi import Depends, FastAPI, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from jose import JWTError, jwt
from passlib.context import CryptContext
from pydantic import BaseModel




class CropModel(BaseModel):
    """
    Crop Model
    fields: N,P,K,temperature,humidity,ph,rainfall
    """
    N: float
    P: float
    K: float
    temperature: float
    humidity: float
    ph: float
    rainfall: float


class CropPrediction(BaseModel):
    """
    Crop Prediction
    fields: crop
    """
    crop: str


class FertilizerModel(BaseModel):
    """
    Fertilizer Model
    fields: Temparature,Humidity ,Moisture,Soil Type,Crop Type,Nitrogen,Potassium,Phosphorous

    """
    Temparature: float
    Humidity: float
    Moisture: float
    Soil_Type: str
    Crop_Type: str
    Nitrogen: float
    Potassium: float
    Phosphorous: float


class FertilizerPrediction(BaseModel):
    """
    Fertilizer Prediction
    fields: Fertilizer Name
    """
    Fertilizer_Name: str


class RainFallModel(BaseModel):
    """
    RainFall Model
    fields: "avg.temp","max.temp","min.temp","precipitation","avg.wind.speed","max.wind.speed","max.wind.speed.dir","max.inst.wind.speed","max.inst.wind.speed.dir","min.atmos.pressure"
    """
    avgTemp = float
    maxTemp = float
    minTemp = float
    precipitation = float
    avgWindSpeed = float
    maxWindSpeed = float
    maxWindSpeedDir = float
    maxInstWindSpeed = float
    maxInstWindSpeedDir = float
    minAtmosPressure = float


class RainFallPrediction(BaseModel):
    """
    RainFall Prediction
    fields: RainFall(N: No rain L: Light rain H: Heavy rain)
    """
    RainFall = str



class Token(BaseModel):
    access_token: str
    token_type: str


class TokenData(BaseModel):
    username: Union[str, None] = None


class User(BaseModel):
    username: str
    email: Union[str, None] = None
    full_name: Union[str, None] = None
    disabled: Union[bool, None] = None


class UserInDB(User):
    hashed_password: str
