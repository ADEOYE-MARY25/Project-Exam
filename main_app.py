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
model= joblib.load("model.pkl")
scalar=joblib.load("scaler.pkl")


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
        volatile_acidity: float
        alcohol: float
      
       


@app.get("/")
def home():
        return{'message', 'welcome to Wine dataset Predictor'}

@app.post("/predict_wine_quality")
def get_predicted_quality(input: Data):
    features=np.array([[
        input.fixed_acidity,
        input.citric_acid,
        input.residual_sugar, 
        input.chlorides,
        input.free_sulfur_dioxide,
        input.total_sulfur_dioxide,
        input.density,
        input.pH,
        input.sulphates,
        input.volatile_acidity,
        input.alcohol
    ]])
    
    x_scaled= scalar.fit_transform(features)
    y_prediction = model.predict(x_scaled)
    return (y_prediction[0])
        

