from typing import Any, Dict, List, Tuple, Union

import numpy as np
from sklearn.pipeline import Pipeline

from automl_common.ensemble_building.abstract_ensemble import AbstractEnsemble


class MockEnsemble(AbstractEnsemble):
    """Implements the abstract class just so it can be saved and loaded"""

    def __init__(self, id: int):
        self.id = id

    def fit(
        self,
        base_model_predictions: np.ndarray,
        true_targets: np.ndarray,
        model_identifiers: List[Tuple[int, int, float]],
    ) -> "MockEnsemble":
        return self

    def predict(
        self, base_model_predictions: Union[np.ndarray, List[np.ndarray]]
    ) -> np.ndarray:
        return np.array([])

    def get_models_with_weights(
        self, models: Dict[str, Pipeline]
    ) -> List[Tuple[float, Pipeline]]:
        return []

    def get_selected_model_identifiers(self) -> List[Tuple[int, int, float]]:
        return []

    def get_validation_performance(self) -> float:
        return 0.0

    def __eq__(self, other: Any):
        if not isinstance(other, AbstractEnsemble):
            raise NotImplementedError()

        return self.id == other.id
