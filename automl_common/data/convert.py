from typing import Union, List
import numpy as np


def probabilities_to_classes(
    probabilities: np.ndarray,
    classes: Union[np.ndarray, List[np.ndarray]],
) -> np.ndarray:
    """Convert probabilities to classes

    Using code from DummyClassifier `fit` and `predict`
    https://github.com/scikit-learn/scikit-learn/blob/7e1e6d09bcc2eaeba98f7e737aac2ac782f0e5f1/sklearn/dummy.py#L307
    """  # noqa: E501
    shape = np.shape(classes)
    if len(shape) == 1:
        n_outputs = 1
        classes = [classes]
        probabilities = [probabilities]

    elif len(shape) == 2:
        n_outputs = len(classes)
    else:
        raise NotImplementedError(f"Don't support `classes` with ndim > 2, {classes}")

    predictions = np.vstack(
        [
            classes[k][probabilities[k].argmax(axis=1)]
            for k in range(n_outputs)
        ]
    ).T

    return predictions
