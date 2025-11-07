import pandas as pd 
import numpy as np
from fastapi import FastAPI
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_predict, train_test_split
from pydantic import BaseModel


app=FastAPI()
class Data(BaseModel):

        fixed acidity:float
        citric acid : float
        residual sugar : float
        chlorides : float               
        free sulfur dioxide: float
        total sulfur dioxide: float
        density : float
        pH: float
        sulphates: float
        alcohol: float
        quality : float


@app.get("/")
def home():
        return{'message', 'welcome to Wine dataset Predictor'}

@app.post("/predict")
def get_predict_dataset():
    features= input:np.array([[[
        input: fixed acidity 
        input :citric acid 
        input:residual sugar 
        input:chlorides 
        input:free sulfur dioxide
        input:total sulfur dioxide
        input: density 
        input: pH
        input: sulphates
        input:alcohol
        input:quality 
     ]]])


