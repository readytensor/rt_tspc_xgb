import os
import warnings

import joblib
import numpy as np
from xgboost import XGBClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.exceptions import NotFittedError
from multiprocessing import cpu_count
from sklearn.metrics import f1_score
from schema.data_schema import TSAnnotationSchema
from preprocessing.custom_transformers import PADDING_VALUE
from typing import Optional, Tuple

warnings.filterwarnings("ignore")
PREDICTOR_FILE_NAME = "predictor.joblib"

# Determine the number of CPUs available
n_cpus = cpu_count()

# Set n_jobs to be one less than the number of CPUs, with a minimum of 1
n_jobs = max(1, n_cpus - 1)
print(f"Using n_jobs = {n_jobs}")


class TSAnnotator:
    """XGBoost Timeseries Annotator.

    This class provides a consistent interface that can be used with other
    TSAnnotator models.
    """

    MODEL_NAME = "XGBoost_Timeseries_Annotator"

    def __init__(
        self,
        data_schema: TSAnnotationSchema,
        encode_len: int,
        n_estimators: int = 100,
        max_depth: int = None,
        min_samples_split: int = 2,
        eta: Optional[float] = 0.3,
        gamma: Optional[float] = 0.0,
        **kwargs,
    ):
        """
        Construct a new XGBoost TSAnnotator.

        Args:
            data_schema (TSAnnotationSchema): The data schema.
            encode_len (int): Encoding (history) length.
            n_estimators (int): Number of trees in the forest.
            max_depth (int): Maximum depth of the tree.
            min_samples_split (int): Minimum number of samples required to split an internal node.
            eta (float): Learning rate.
            gamma (float): Minimum loss reduction required to make a further partition on a leaf node.
        """
        self.data_schema = data_schema
        self.encode_len = int(encode_len)
        self.n_estimators = int(n_estimators)
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.eta = eta
        self.gamma = gamma
        self.kwargs = kwargs
        self.model = self.build_model()
        self._is_trained = False

    def build_model(self) -> XGBClassifier:
        """Build a new XGBoost Classifier."""
        model = XGBClassifier(
            multi_strategy='multi_output_tree',
            objective='multi:softmax',
            max_depth=self.max_depth,
            n_estimators=self.n_estimators,
            eta=self.eta,
            gamma=self.gamma,
            num_class=len(self.data_schema.target_classes),
            n_jobs=n_jobs,
            **self.kwargs,
        )
        return MultiOutputClassifier(model, n_jobs=n_jobs)

    def _get_X_and_y(
        self, data: np.ndarray, is_train: bool = True
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Extract X (historical target series), y (forecast window target)
        When is_train is True, data contains both history and forecast windows.
        When False, only history is contained.
        """
        N, T, D = data.shape
        if is_train:
            if T != self.encode_len:
                raise ValueError(
                    f"Training data expected to have {self.encode_len}"
                    f" length on axis 1. Found length {T}"
                )
            # we excluded the first 2 dimensions (id, time) and the last dimension (target)
            X = data[:, :, 2:-1].reshape(N, -1)  # shape = [N, T*D]
            y = data[:, :, -1].astype(int).reshape(N, -1)  # shape = [N, T]
        else:
            # for inference
            if T < self.encode_len:
                raise ValueError(
                    f"Inference data length expected to be >= {self.encode_len}"
                    f" on axis 1. Found length {T}"
                )
            # X = data.reshape(N, -1)
            X = data[:, :, 2:].reshape(N, -1)
            y = data[:, :, 0:2]
        return X, y

    def fit(self, train_data):
        train_X, train_y = self._get_X_and_y(train_data, is_train=True)
        self.model.fit(train_X, train_y)
        self._is_trained = True
        return self.model

    def predict(self, data):
        X, window_ids = self._get_X_and_y(data, is_train=False)
        preds = self.model.predict_proba(X)
        for i in range(len(preds)):
            if preds[i].shape[0] > len(self.data_schema.target_classes):
                preds[i] = preds[i][:-1]
        preds = np.array(preds)
        preds = preds.transpose(1, 0, 2)

        prob_dict = {}

        for index, prediction in enumerate(preds):
            series_id = window_ids[index][0][0]
            for step_index, step in enumerate(prediction):
                step_id = window_ids[index][step_index][1]
                step_id = (series_id, step_id)
                prob_dict[step_id] = prob_dict.get(step_id, []) + [step]

        prob_dict = {
            k: np.mean(np.array(v), axis=0)
            for k, v in prob_dict.items()
            if k[1] != PADDING_VALUE
        }

        sorted_dict = {key: prob_dict[key] for key in sorted(prob_dict.keys())}
        probabilities = np.vstack(sorted_dict.values())
        return probabilities

    def evaluate(self, test_data):
        """Evaluate the model and return the loss and metrics"""
        x_test, y_test = self._get_X_and_y(test_data, is_train=True)
        if self.model is not None:
            prediction = self.model.predict(x_test).flatten()
            y_test = y_test.flatten()
            f1 = f1_score(y_test, prediction, average="weighted")
            return f1

        raise NotFittedError("Model is not fitted yet.")

    def save(self, model_dir_path: str) -> None:
        """Save the XGboost TSAnnotator to disk.

        Args:
            model_dir_path (str): Dir path to which to save the model.
        """
        if not self._is_trained:
            raise NotFittedError("Model is not fitted yet.")
        joblib.dump(self, os.path.join(model_dir_path, PREDICTOR_FILE_NAME))

    @classmethod
    def load(cls, model_dir_path: str) -> "TSAnnotator":
        """Load the XGboost TSAnnotator from disk.

        Args:
            model_dir_path (str): Dir path to the saved model.
        Returns:
            TSAnnotator: A new instance of the loaded XGBoost TSAnnotator.
        """
        model = joblib.load(os.path.join(model_dir_path, PREDICTOR_FILE_NAME))
        return model


def train_predictor_model(
    train_data: np.ndarray,
    data_schema: TSAnnotationSchema,
    hyperparameters: dict,
) -> TSAnnotator:
    """
    Instantiate and train the TSAnnotator model.

    Args:
        train_data (np.ndarray): The train split from training data.
        hyperparameters (dict): Hyperparameters for the TSAnnotator.

    Returns:
        'TSAnnotator': The TSAnnotator model
    """
    model = TSAnnotator(
        data_schema=data_schema,
        **hyperparameters,
    )
    model.fit(train_data=train_data)
    return model


def predict_with_model(model: TSAnnotator, test_data: np.ndarray) -> np.ndarray:
    """
    Make forecast.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_data (np.ndarray): The test input data for annotation.

    Returns:
        np.ndarray: The annotated data.
    """
    return model.predict(test_data)


def save_predictor_model(model: TSAnnotator, predictor_dir_path: str) -> None:
    """
    Save the TSAnnotator model to disk.

    Args:
        model (TSAnnotator): The TSAnnotator model to save.
        predictor_dir_path (str): Dir path to which to save the model.
    """
    if not os.path.exists(predictor_dir_path):
        os.makedirs(predictor_dir_path)
    model.save(predictor_dir_path)


def load_predictor_model(predictor_dir_path: str) -> TSAnnotator:
    """
    Load the TSAnnotator model from disk.

    Args:
        predictor_dir_path (str): Dir path where model is saved.

    Returns:
        TSAnnotator: A new instance of the loaded TSAnnotator model.
    """
    return TSAnnotator.load(predictor_dir_path)


def evaluate_predictor_model(model: TSAnnotator, test_split: np.ndarray) -> float:
    """
    Evaluate the TSAnnotator model and return the r-squared value.

    Args:
        model (TSAnnotator): The TSAnnotator model.
        test_split (np.ndarray): Test data.

    Returns:
        float: The r-squared value of the TSAnnotator model.
    """
    return model.evaluate(test_split)
