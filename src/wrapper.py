"""
Wrapper for the XGBoost model.
"""
import pickle
import pandas as pd
from src.dtos import OnePredictionOutputDto, FilePredictionOutputDto

FEATURES_TO_ENCODE: list[str] = ['hotel', 'arrival_date_month', 'meal', 'country', 'market_segment',
                                 'distribution_channel', 'is_repeated_guest', 'reserved_room_type',
                                 'assigned_room_type', 'deposit_type', 'customer_type']


class XGBoostModelWrapper:
    def __init__(self, model_path: str, model_encoder_path: str):
        # create the instance of the model
        self.model = self.load_model(model_path)

        # load the OneHotEncoder model
        self.one_hot_encoder = self.load_encoder(model_encoder_path)

    @staticmethod
    def load_model(model_path: str):
        """
        Load the XGBoost model

        Args:
            model_path (str): Path to the model.

        Returns:
            model: XGBoost model.

        Raise:
            RuntimeError: If there is an error loading the model.
        """

        try:
            with open(model_path, 'rb') as f:
                model = pickle.load(f)

            return model
        except Exception:
            raise RuntimeError(f"Error loading the model.")

    @staticmethod
    def load_encoder(model_encoder_path: str):
        """
        Load the One-Hot-Encoder.

        Args:
            model_encoder_path (str): Path to the one hot encoder model.

        Returns:
            one_hot_encoder: The loaded one hot encoder.

        Raise:
            RuntimeError: If there is an error loading the encoder.
        """

        try:

            with open(model_encoder_path, 'rb') as f:
                one_hot_encoder = pickle.load(f)

                return one_hot_encoder

        except Exception:
            raise RuntimeError(f"Error loading the encoder")

    @staticmethod
    def preprocess_data(df: pd.DataFrame, one_hot_encoder) -> pd.DataFrame:
        """
        Preprocess the input data, apply the OneHotEncoder to the categorical features and return the dataframe.

        Args:
            df (pd.DataFrame): The input dataframe to preprocess.
            one_hot_encoder: The OneHotEncoder object to use for preprocessing.

        Returns:
            pd.DataFrame: The preprocessed dataframe with the categorical features encoded.

        Raise:
            ValueError: If there is an error preprocessing the input data.
        """

        try:
            # do the preprocessing of the input ------------------------------------------------------------------------

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
            raise ValueError("Error in preprocessing the input data. Please check the input file.")

    @staticmethod
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

    def predict(self, df: pd.DataFrame):
        """
        Do the preprocessing and the predictions.

        Args:
            df (pd.DataFrame): The elements to do the prediction on.

        Returns:
            The made predictions.
        """

        try:
            # 1. preprocessing of the data
            # encode the data using the One-Hot-Encoder
            df_encoded = self.preprocess_data(df=df, one_hot_encoder=self.one_hot_encoder)

            # 2. do the prediction
            pred = self.model.predict(df_encoded)
            pred_proba = self.model.predict_proba(df_encoded)

            return pred, pred_proba

        except Exception:
            raise ValueError(f"Oops, error while doing the prediction.")

    def predict_one(self, data: dict) -> OnePredictionOutputDto:
        """
        Do the predictions for just one set of features.

        Args:
            data (dict): The features as dictionary to do the prediction.

        Returns:
            dict: The prediction as a dictionary.
        """

        try:
            # transform from dict to df
            df = self.from_dict_to_df(data)

            # do the predictions
            pred, pred_proba = self.predict(df)

            # create the output transform
            output: dict = {
                "prediction": pred.tolist(),
                "proba_0": pred_proba.tolist()[0][0],
                "proba_1": pred_proba.tolist()[0][1]
            }

            return output

        except Exception:
            raise ValueError(f"Oops, error while doing the prediction.")

    def predict_from_file(self, file) -> FilePredictionOutputDto:
        """
        Do the predictions for just one set of features.

        Args:
            file (file): The input file of all the features to do the predictions.

        Returns:
            dict: The predictions as a dictionary.
        """

        try:
            # transform from dict to df
            df = pd.read_csv(file.file)

            # verify if the file has the client in the columns
            client_info_df = None

            if 'client' in df.columns:
                # info of the cliente to save
                client_info_columns = [
                    'client', 'country', 'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'
                ]

                # get the client info
                client_info_df = df[client_info_columns]

                # crate the date of the reservation
                client_info_df['reservation_date'] = pd.to_datetime(
                    client_info_df['arrival_date_year'].astype(str) + '-' +
                    client_info_df['arrival_date_month'] + '-' +
                    client_info_df['arrival_date_day_of_month'].astype(str)
                )

                # drop the columns that are not needed now
                client_info_df = client_info_df.drop(
                    ['arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month'],
                    axis=1
                )

                # clean the df
                df = df.drop(
                    'client',
                    axis=1
                )

            # do the predictions
            predictions, predictions_probas = self.predict(df)

            # get the probabilities for the classes
            predictions_to_return = list(
                map(
                    lambda x, y: {'label': x, 'proba_0': y[0], 'proba_1': y[1]},
                    predictions.tolist(),
                    predictions_probas.tolist()
                )
            )

            # if there were clients, add them to the dataframe
            if client_info_df is not None:
                # add the prediction to the client info
                client_info_df['is_going_to_cancel'] = predictions

                # add the probability predictions to the client info
                client_info_df['prob_cancel'] = list(map(lambda x: x[1], predictions_probas.tolist()))
                client_info_df['prob_not_cancel'] = list(map(lambda x: x[0], predictions_probas.tolist()))

            # create the output transform
            output: dict = {
                "predictions": predictions_to_return,
                "client_info": client_info_df.to_dict(orient='records') if client_info_df is not None else None,
                "total_predictions": len(predictions),
                "file_name": file.filename
            }

            return output

        except Exception as e:
            print(e)
            raise ValueError(f"Oops, error while doing the predictions.")
