from __future__ import annotations

from typing import Dict, List, Tuple, TypeVar

from sklearn.utils.validation import check_is_fitted

from automl_common.sklearn.ensemble.base import Ensemble
from automl_common.sklearn.model import Predictor
from automl_common.util.types import Orderable

PredictorT = TypeVar("PredictorT", bound=Predictor)


class WeightedEnsemble(Ensemble[PredictorT]):
    """An ensemble of size ``n`` that selects a weighted ensemble."""

    @property
    def trajectory(self) -> List[Tuple[str, Orderable]]:
        """The trajectory of the fitting procedue

        Returns
        -------
        List[Tuple[str, Orderable]]
            The trajectory of the ensemble fitting procedure with the model added and
            the overall ensemble performance with that model added.

            Note
            ----


            .. code-block:: python

                for model_id, perf in ensemble.trajectory:
                    ...

        Raises
        ------
        NotFittedError
            If the ensemble has not been fitted yet
        """
        check_is_fitted(self)
        return self.trajectory_  # type: ignore

    @property
    def weights(self) -> Dict[str, float]:
        """The weights of the ensemble

        Returns
        -------
        Dict[str, float]
            A dictionary mapping from model ids to weights

            .. code-block:: python

                {
                    "model_a": 0.6,
                    "model_b": 0.2,
                    "model_c": 0.2,
                }

        Raises
        ------
        NotFittedError
            If the ensemble has not been fit yet
        """
        check_is_fitted(self)
        return self.weights_

    @classmethod
    def _fit_attributes(cls) -> List[str]:
        """The attributes required by this ensemble.

        Implementing classes should take into account their superclasses attributes

        .. code-block:: python

            @classmethod
            def _attributes(cls) -> List[str]:
                return super()._attributes() + [...]

        Returns
        -------
        List[str]
            The attributes of this ensemble
        """
        return super()._fit_attributes() + ["weights_", "trajectory_"]
