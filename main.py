# Put the code for your API here.

from fastapi import FastAPI
from pydantic import BaseModel, Field
from typing import Literal
import pickle as pkl
import numpy as np
import pandas as pd

app = FastAPI()

# Load model and preprocessors
with open("model/decision_tree.pkl", "rb") as model_file:
    model = pkl.load(model_file)
with open("model/encoder.pkl", "rb") as encoder_file:
    encoder = pkl.load(encoder_file)
with open("model/lb.pkl", "rb") as lb_file:
    lb = pkl.load(lb_file)


# Define input schema
class InferenceInput(BaseModel):
    age: int = Field(..., example=39)
    workclass: str = Field(..., example="State-gov")
    fnlgt: int = Field(..., example=77516)
    education: str = Field(..., example="Bachelors")
    education_num: int = Field(..., alias="education-num", example=13)
    marital_status: str = Field(..., alias="marital-status", example="Never-married")
    occupation: str = Field(..., example="Adm-clerical")
    relationship: str = Field(..., example="Not-in-family")
    race: str = Field(..., example="White")
    sex: Literal["Male", "Female"] = Field(..., example="Male")
    capital_gain: int = Field(..., alias="capital-gain", example=2174)
    capital_loss: int = Field(..., alias="capital-loss", example=0)
    hours_per_week: int = Field(..., alias="hours-per-week", example=40)
    native_country: str = Field(..., alias="native-country", example="United-States")

    class Config:
        schema_extra = {
            "example": {
                "age": 39,
                "workclass": "State-gov",
                "fnlgt": 77516,
                "education": "Bachelors",
                "education-num": 13,
                "marital-status": "Never-married",
                "occupation": "Adm-clerical",
                "relationship": "Not-in-family",
                "race": "White",
                "sex": "Male",
                "capital-gain": 2174,
                "capital-loss": 0,
                "hours-per-week": 40,
                "native-country": "United-States"
            }
        }


@app.get("/")
def read_root():
    return {"message": "Welcome to the Census Income Prediction API"}

@app.post("/predict")
def predict(input_data: InferenceInput):
    data_dict = input_data.dict(by_alias=True)

    from starter.ml.data import process_data

    data_dict = input_data.dict(by_alias=True)
    data_df = pd.DataFrame([data_dict])  # Wrap in list to create a single-row DataFrame

    X, _, _, _ = process_data(
        data_df,
        categorical_features=[
            "workclass", "education", "marital-status", "occupation",
            "relationship", "race", "sex", "native-country"
        ],
        label=None,
        training=False,
        encoder=encoder,
        lb=lb
    )

    prediction = model.predict(X)
    pred_label = lb.inverse_transform(prediction)[0]
    return {"prediction": pred_label}