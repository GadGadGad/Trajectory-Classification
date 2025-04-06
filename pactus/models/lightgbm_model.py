from typing import Any, List, Union

import numpy as np
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder
from lightgbm import LGBMClassifier
from yupi import Trajectory

from pactus import featurizers
from pactus.dataset import Data
from pactus.models.model import Model

NAME = "lightgbm"


class LightGBMModel(Model):
    """Implementation of a LightGBM Classifier."""

    def __init__(self, featurizer: featurizers.Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.model = LGBMClassifier(**kwargs)
        self.encoder: Union[LabelEncoder, None] = None
        self.grid: GridSearchCV
        self.set_summary(**kwargs)

    def _encode_labels(self, data: Data) -> np.ndarray:
        """Encode the labels"""
        if self.encoder is None:
            self.encoder = LabelEncoder()
            self.encoder.fit(data.labels)
        encoded_labels = self.encoder.transform(data.labels)
        assert isinstance(encoded_labels, np.ndarray)
        return encoded_labels

    def train(self, data: Data, cross_validation: int = 0, grid_params: dict = {}):
        self.set_summary(cross_validation=cross_validation)
        x_data = data.featurize(self.featurizer)
        self.grid = GridSearchCV(self.model, grid_params, cv=cross_validation, verbose=3)
        classes = self._encode_labels(data)
        self.grid.fit(x_data, classes)

    def predict(self, data: Data) -> List[Any]:
        x_data = data.featurize(self.featurizer)
        predicted = self.grid.predict(x_data)
        assert self.encoder is not None
        return self.encoder.inverse_transform(predicted)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.grid.predict([traj])[0]
