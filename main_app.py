import pandas as pd 
import numpy as np
from fastapi import FastAPI
import joblib
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_predict, train_test_split, RandomizedSearchCV
from pydantic import BaseModel
from random import randint


app=FastAPI()
model= joblib.load("../model/model.pkl")
scalar=joblib.load("../model/scalar.pkl")


class Data(BaseModel):
        fixed_acidity:float
        citric_acid : float
        residual_sugar : float
        chlorides : float               
        free_sulfur_dioxide: float
        total_sulfur_dioxide: float
        density : float
        pH: float
        sulphates: float
        alcohol: float
        quality : float


@app.get("/")
def home():
        return{'message', 'welcome to Wine dataset Predictor'}

@app.post("/predict_wine_quality")
def get_predicted_quality():
    features=np.array([[
        input.fixed acidity,
        input.citric acid,
        input.residual sugar, 
        input.chlorides,
        input.free_sulfur_dioxide,
        input.total_sulfur_dioxide,
        input.density,
        input. pH,
        input. sulphates,
        input.alcohol
    ]])
    
    x_scaled= scalar.fit_transform(features)
    y_prediction = model.predic(x_scaled)
    return (y_prediction[0])
        

