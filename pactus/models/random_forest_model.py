from typing import Any, List

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from yupi import Trajectory

from pactus import featurizers
from pactus.dataset import Data
from pactus.models import Model

NAME = "random_forest"


class RandomForestModel(Model):
    """Implementation of a Random Forest Classifier."""

    def __init__(self, featurizer: featurizers.Featurizer, **kwargs):
        super().__init__(NAME)
        self.featurizer = featurizer
        self.model = RandomForestClassifier(**kwargs)
        self.grid: GridSearchCV
        self.set_summary(**kwargs)

    def train(self, data: Data, cross_validation: int = 0, param_grid: dict = {}):
        self.set_summary(cross_validation=cross_validation)
        x_data = data.featurize(self.featurizer)
        self.grid = GridSearchCV(self.model, {}, cv=cross_validation, verbose=3)
        self.grid.fit(x_data, data.labels)

    def predict(self, data: Data) -> List[Any]:
        x_data = data.featurize(self.featurizer)
        return self.grid.predict(x_data)

    def predict_single(self, traj: Trajectory) -> Any:
        """Predicts the label of a single trajectory."""
        return self.grid.predict([traj])[0]
