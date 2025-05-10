import pickle
import pandas as pd

from fastapi import FastAPI, UploadFile, HTTPException
from pprint import pprint

from src.dtos import FeaturesDto
from src.utils import preprocess_data, from_dict_to_df


# load the ML model
model_path = "models/opt_model.pkl"

with open(model_path, 'rb') as f:
    model = pickle.load(f)

# load the OneHotEncoder model
one_hot_encoder_path = "models/one_hot_encoder.pkl"

with open(one_hot_encoder_path, 'rb') as f:
    one_hot_encoder = pickle.load(f)


app = FastAPI()


@app.get("/")
def read_root():
    return {"status": "Hi there! I'm a classification API."}


@app.post("/predict/")
async def predict_from_file(file: UploadFile):
    """
    Endpoint for doing batch processing of a .csv file.

    Args:
        file (UploadFile): The input file to process.
    """

    # load the file as csv ---------------------------------------------------------------------------------------------
    try:
        df = pd.read_csv(file.file)
    except:
        raise HTTPException(
            status_code=400,
            detail="Error loading the file. Please check the format of the .cvs file."
        )

    # do the preprocessing of the input --------------------------------------------------------------------------------
    df_encoded = preprocess_data(df=df, one_hot_encoder=one_hot_encoder)

    # do the prediction ------------------------------------------------------------------------------------------------
    try:
        result = model.predict(df_encoded)
    except:
        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )

    # return the predictions -------------------------------------------------------------------------------------------
    return {
        "predictions": result.tolist(),
        "total_predictions": len(result),
        "file_name": file.filename
    }


@app.post("/predict")
def predict(features_dto: FeaturesDto):
    """
    Endpoint for doing one prediction.

    Args:
        features_dto (FeaturesDto): The features to do a prediction.
    """

    # get the data from the request as a dictionary
    features: dict = features_dto.model_dump()

    # transform the features from a dictionary to a pandas dataframe
    df = from_dict_to_df(features)

    # do the preprocessing of the input --------------------------------------------------------------------------------
    df_encoded = preprocess_data(df=df, one_hot_encoder=one_hot_encoder)

    # do the predictions -----------------------------------------------------------------------------------------------
    try:
        pred = model.predict(df_encoded)
        pred_proba = model.predict_proba(df_encoded)
    except Exception as e:
        pprint(e)

        raise HTTPException(
            status_code=500,
            detail="Error doing the prediction."
        )

    # return the predictions -------------------------------------------------------------------------------------------
    return {
        "prediction": pred.tolist(),
        "proba_0": pred_proba.tolist()[0][0],
        "proba_1": pred_proba.tolist()[0][1]
    }
