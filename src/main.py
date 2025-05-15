import pandas as pd
from fastapi import FastAPI, UploadFile, HTTPException
from src.dtos import FeaturesDto
from src.wrapper import XGBoostModelWrapper

# load the ML model
try:
    model_wrapper = XGBoostModelWrapper(
        model_path="models/opt_model.pkl",
        model_encoder_path="models/one_hot_encoder.pkl"
    )
except RuntimeError as e:

    raise Exception(f"NoModel cannot be initialized {e}")

app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "Hi there! I'm a classification API."}


@app.post("/predict-from-file")
async def predict_from_file(file: UploadFile):
    """
    Endpoint for doing batch processing of a .csv file.

    Args:
        file (UploadFile): The input file to process.
    """

    try:
        result = model_wrapper.predict_from_file(file)

        return result
    except:
        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )


@app.post("/predict")
def predict(features_dto: FeaturesDto):
    """
    Endpoint for doing one prediction.

    Args:
        features_dto (FeaturesDto): The features to do a prediction.
    """

    try:
        # get the data from the request as a dictionary
        features: dict = features_dto.model_dump()

        result = model_wrapper.predict_one(features)

        return result
    except:
        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )
