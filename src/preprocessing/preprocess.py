import os
from typing import Any, Dict, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline


from preprocessing.pipeline import (
    create_preprocess_pipelines,
    train_pipeline,
    transform_inputs,
    save_pipeline,
    load_pipeline,
)


TRAINING_PIPELINE_FILE_NAME = "training_pipeline.joblib"
INFERENCE_PIPELINE_FILE_NAME = "inference_pipeline.joblib"


def get_preprocessing_pipelines(
    data_schema: Any,
    preprocessing_config: Dict,
    default_hyperparameters: Dict,
) -> Tuple[Pipeline, Pipeline, int]:
    """
    Get training and inference preprocessing pipeline.

    Args:
        data_schema (Any): A dictionary containing the data schema.
        data (pd.DataFame): A pandas DataFrame containing the train data split.
        preprocessing_config (Dict): A dictionary containing the preprocessing params.
        default_hyperparameters (Dict): A dictionary containing hyperparameters.

    Returns:
        A tuple containing:
            training_pipeline (Pipeline)
            inference_pipeline (Pipeline)
            encoding length (int)
    """
    encode_len = default_hyperparameters["encode_len"]
    # create training and inference pipelines
    training_pipeline, inference_pipeline = create_preprocess_pipelines(
        data_schema=data_schema,
        preprocessing_config=preprocessing_config,
        encode_len=encode_len,
    )
    return training_pipeline, inference_pipeline


def fit_transform_with_pipeline(
    pipeline: Pipeline, data: pd.DataFrame
) -> Tuple[Pipeline, np.ndarray]:
    """
    Fit the preprocessing pipeline and transform data.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The data as a numpy array

    Returns:
        Pipeline: Fitted preprocessing pipeline.
        np.ndarray: transformed data as a numpy array.
    """
    trained_pipeline = train_pipeline(pipeline, data)
    transformed_data = transform_inputs(trained_pipeline, data)
    return trained_pipeline, transformed_data


def fit_pipeline(pipeline: Pipeline, data: pd.DataFrame) -> Pipeline:
    """
    Fit the preprocessing pipeline.

    Args:
        pipeline (Pipeline): The preprocessing pipeline.
        data (pd.DataFrame): The data as a numpy array

    Returns:
        Pipeline: Fitted preprocessing pipeline.
    """
    return train_pipeline(pipeline, data)


def transform_data(
    preprocess_pipeline: Any, data: pd.DataFrame
) -> Tuple[pd.DataFrame, Union[pd.Series, None]]:
    """
    Transform the data using the preprocessing pipeline and target encoder.

    Args:
        preprocess_pipeline (Any): The preprocessing pipeline.
        target_encoder (Any): The target encoder.
        data (pd.DataFrame): The input data as a DataFrame (targets may be included).

    Returns:
        Tuple[pd.DataFrame, Union[pd.Series, None]]: A tuple containing the transformed
            data and transformed targets;
            transformed targets are None if the data does not contain targets.
    """
    transformed_inputs = transform_inputs(preprocess_pipeline, data)
    return transformed_inputs


def get_encode_len(
    train_data, data_schema, encode_to_decode_ratio, min_windows, min_encode_len_ratio
):
    """
    Calculate the length of the encoding (history) window for time series forecasting,
    given constraints on minimum window size, minimum number of windows, and the
    ratio of encoding to decoding (forecast) length.

    The function determines the optimal length of the history window used for training
    the forecasting model. This length is constrained by the total available history,
    the desired forecast period, the minimum ratio of history window to forecast window,
    and the minimum number of sliding windows desired for model training.

    Args:
    - train_data (DataFrame): The dataset containing the time series data.
    - data_schema (object): An object with metadata about the dataset, including:
      - time_col (str): Name of the column in train_data representing time.
      - forecast_length (int): The length of the forecast period.
    - encode_to_decode_ratio (float): The maximum allowed ratio of the length of the
      encoding window to the decoding (forecast) window length.
    - min_encode_len_ratio (float): The minimum ratio of the encoding window length to
                                    the decoding window length.
                                    This is a hard minimum.
    - min_windows (int): The minimum number of sliding windows desired for training.
                         Note that this is a "soft" minimum. The resulting number of
                         windows might be lower if history is shorter than required.

    Returns:
    - int: The calculated length of the encoding window.

    Raises:
    - ValueError: If the total history length is not sufficient for the minimum
      required encoding and decoding lengths plus the minimum number of windows.

    The function first calculates the minimum required encoding length based on
    the min_encode_len_ratio and ensures that the total history is sufficient to
    create at least 1 window by using the minimum encoding length plus the
    forecast period.
    It then keeps the encoding length constant as history length increases
    which will increase the number of windows. If given enough history to achieve the
    minimum number of windows, it then increases the encoding length, up to a
    maximum of  encode_to_decode_ratio * forecast_length.
    """
    history_len = train_data[data_schema.time_col].nunique()
    decode_len = data_schema.forecast_length
    min_encode_len = max(1, int(decode_len * min_encode_len_ratio))

    if history_len < decode_len + min_encode_len:
        raise ValueError(
            f"History length ({history_len}) must be at least forecast length"
            f" ({decode_len}) + min_encode_len ({min_encode_len})"
        )

    available_len = history_len - decode_len - min_windows + 1
    if available_len < min_encode_len:
        encode_len = min_encode_len
    else:
        encode_len = available_len
    if encode_len > int(encode_to_decode_ratio * decode_len):
        encode_len = int(encode_to_decode_ratio * decode_len)
    return encode_len


def save_pipelines(
    training_pipeline: Any, inference_pipeline: Any, preprocessing_dir_path: str
) -> None:
    """
    Save the preprocessing pipeline and target encoder to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        target_encoder: The target encoder.
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            are saved
    """
    if not os.path.exists(preprocessing_dir_path):
        os.makedirs(preprocessing_dir_path)
    save_pipeline(
        pipeline=training_pipeline,
        file_path_and_name=os.path.join(
            preprocessing_dir_path, TRAINING_PIPELINE_FILE_NAME
        ),
    )
    save_pipeline(
        pipeline=inference_pipeline,
        file_path_and_name=os.path.join(
            preprocessing_dir_path, INFERENCE_PIPELINE_FILE_NAME
        ),
    )


def save_preprocessing_pipeline(
    preprocess_pipeline: Any, preprocessing_dir_path: str, pipeline_type: str
) -> None:
    """
    Save the preprocessing pipeline to files.

    Args:
        preprocess_pipeline: The preprocessing pipeline.
        preprocessing_dir_path (str): dir path where the pipeline is to be saved.
        pipeline_type (str): Type of pipeline to be saved. Can be either "training" or
                            "inference"

    """
    if not os.path.exists(preprocessing_dir_path):
        os.makedirs(preprocessing_dir_path)

    if pipeline_type == "training":
        file_name = TRAINING_PIPELINE_FILE_NAME
    else:
        file_name = INFERENCE_PIPELINE_FILE_NAME
    save_pipeline(
        pipeline=preprocess_pipeline,
        file_path_and_name=os.path.join(preprocessing_dir_path, file_name),
    )


def load_pipeline_of_type(preprocessing_dir_path: str, pipeline_type: str) -> Pipeline:
    """
    Load the preprocessing pipeline and target encoder

    Args:
        preprocessing_dir_path (str): dir path where the pipeline and target encoder
            are saved
        pipeline_type (str): Type of pipeline to be loaded. Can be either "training" or
                            "inference"

    Returns:
        Pipeline: Loaded preprocessing pipeline.
    """
    if pipeline_type == "training":
        file_name = TRAINING_PIPELINE_FILE_NAME
    else:
        file_name = INFERENCE_PIPELINE_FILE_NAME
    preprocess_pipeline = load_pipeline(
        file_path_and_name=os.path.join(preprocessing_dir_path, file_name)
    )
    return preprocess_pipeline


def inverse_scale_predictions(
    predictions: np.ndarray, pipeline: Pipeline
) -> np.ndarray:
    """
    Applies the inverse transformation of MinMax scaling to the predictions using the
    trained inference pipeline.

    Args:
        predictions (np.ndarray): The model's predictions, of shape [N, T, D].
        pipeline (Pipeline): The trained inference pipeline containing the MinMax scaler.

    Returns:
        np.ndarray: The inverse-scaled predictions.
    """
    # Retrieve the minmax_scaler from the pipeline
    if "minmax_scaler" in pipeline.named_steps:
        minmax_scaler = pipeline.named_steps["minmax_scaler"]
        return minmax_scaler.inverse_transform(predictions)
    else:
        raise ValueError("MinMaxScaler not found in the pipeline.")