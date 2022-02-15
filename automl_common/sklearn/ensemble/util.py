from typing import Dict, Iterable, Mapping, List, Any

from collections import defaultdict

import numpy as np

from automl_common.util.functional import intersection, union

# https://scikit-learn.org/stable/developers/develop.html#estimator-tags
# A dict mapping from tag -> (what's required for non default, default)
# The defaults aren't used but just here for full documentation
alltags = {
    "allow_nan": (all, False),
    "binary_only": (any, False),
    "multilabel": (all, False),
    "multioutput": (all, False),
    "multioutput_only": (any, False),
    "no_validation": (any, False),
    "non_deterministic": (any, False),
    "pairwise": (any, False),
    "preserves_dtype": (intersection, [np.float64]),
    "poor_score": (all, False),
    "requires_fit": (any, True),
    "requires_positive_X": (any, False),
    "requires_y": (any, False),
    "requires_positive_y": (any, False),
    "_skip_test": (any, False),
    "_xfail_checks": (union, False),  # Can't handle this easily
    "stateless": (all, False),
    "X_types": (intersection, ["2darray"]),
}


def tag_accumulate(model_tags: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Accumulate all model_tags so that the accumulated tags suppports all models

    Parameters
    ----------
    model_tags : Iterable[Mapping[str, Any]]
        An iterable over the tags for each model

    Returns
    -------
    Dict[str, Any]
        The tags which allow all models to be supported
    """
    acctags: Dict[str, List[Any]] = defaultdict(list)

    for tags in model_tags:
        for k, v in tags.items():
            acctags[k].append(v)

    results: Dict[str, Any] = {}
    for k, v in acctags.items():
        accumulation_method, _ = alltags[k]
        results[k] = accumulation_method(v)

    return results
