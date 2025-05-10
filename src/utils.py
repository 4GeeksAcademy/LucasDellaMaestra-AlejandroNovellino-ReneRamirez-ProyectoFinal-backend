"""
Utils functions and constants for the project.
"""

import pandas as pd
from fastapi import HTTPException

# constants
FEATURES_TO_ENCODE: list[str] = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                                 'distribution_channel', 'is_repeated_guest', 'reserved_room_type',
                                 'assigned_room_type', 'deposit_type', 'customer_type']

# functions
def preprocess_data(df: pd.DataFrame, one_hot_encoder) -> pd.DataFrame:
    """
    Preprocess the input data, apply the OneHotEncoder to the categorical features and return the dataframe.

    Args:
        df (pd.DataFrame): The input dataframe to preprocess.
        one_hot_encoder: The OneHotEncoder object to use for preprocessing.

    Returns:
        pd.DataFrame: The preprocessed dataframe with the categorical features encoded.

    Raise:
        HTTPException: If there is an error in preprocessing the input data.
    """

    try:
        # do the preprocessing of the input ----------------------------------------------------------------------------

        # 1. transform the categorical features using the OneHotEncoder
        encoded_features = one_hot_encoder.transform(df[FEATURES_TO_ENCODE])

        # 2. get the features names
        feature_names = one_hot_encoder.get_feature_names_out(FEATURES_TO_ENCODE)

        # 3. transform the encoded features to a pandas dataframe
        encoded_features_df = pd.DataFrame(encoded_features, columns=feature_names, index=df.index)

        # 4. concat the original dataframe with the encoded features
        df_encoded = pd.concat([df.drop(columns=FEATURES_TO_ENCODE), encoded_features_df], axis=1)

        return df_encoded

    except:
        raise HTTPException(
            status_code=400,
            detail="Error in preprocessing the input data. Please check the input file."
        )


def from_dict_to_df(features: dict) -> pd.DataFrame:
    """
    Transform a dictionary to a pandas dataframe.

    Args:
        features (dict): The input dictionary to transform.

    Returns:
        pd.DataFrame: The transformed dataframe.
    """

    # representation of one row
    row: dict = {}

    # for each key and value in the dictionary, we add the key and value to the row
    for key, value in features.items():
        # we add the key and value to the row
        row[key] = value

    # transform to dataframe
    df: pd.DataFrame = pd.DataFrame([row])

    return df
