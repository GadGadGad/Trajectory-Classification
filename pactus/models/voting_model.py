from typing import Any, List, Union

import numpy as np
from sklearn.ensemble import VotingClassifier
from sklearn.preprocessing import LabelEncoder
from yupi import Trajectory

from pactus import featurizers
from pactus.dataset import Data
from pactus.models.model import Model

NAME = "voting_model"


class VotingModel(Model):
    """Implementation of a Voting Classifier combining multiple models."""

    def __init__(self,
                 models: List[Model],
                 voting: str = 'hard',
                 weights: Union[List[float], None] = None,
                 featurizer: featurizers.Featurizer = None,
                 **kwargs):
        super().__init__(NAME)
        self.models = models
        self.voting = voting
        self.weights = weights
        self.encoder: Union[LabelEncoder, None] = None 
        self.ensemble: VotingClassifier
        self.featurizer = featurizer
        self.set_summary(voting=voting, weights=weights, **kwargs)

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
        for model in self.models:
            model.train(data, cross_validation, grid_params) 
    
        estimators = [(model.name, model.grid.best_estimator_) for model in self.models] 
        self.ensemble = VotingClassifier(estimators=estimators, voting=self.voting, weights=self.weights)

        X_data = data.featurize(self.featurizer) 
        classes = self._encode_labels(data)
        self.ensemble.fit(X_data, classes)


    def predict(self, data: Data) -> List[Any]:
        X = data.featurize(self.featurizer)
        return self.ensemble.predict(X)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.ensemble.predict([traj])[0]
