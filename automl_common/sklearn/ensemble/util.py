from typing import Any, Dict, Iterable, List, Mapping

from collections import defaultdict

from automl_common.util.functional import intersection, dictmerge
from sklearn.utils._tags import _DEFAULT_TAGS

# https://scikit-learn.org/stable/developers/develop.html#estimator-tags
# A dict mapping from tag -> (what's required for non default, default)
# The defaults aren't used but just here for full documentation
accumulation_method = {
    "allow_nan": all,
    "binary_only": any,
    "multilabel": all,
    "multioutput": all,
    "multioutput_only": any,
    "no_validation": any,
    "non_deterministic": any,
    "pairwise": any,
    "preserves_dtype": intersection,
    "poor_score": all,
    "requires_fit": any,
    "requires_positive_X": any,
    "requires_y": any,
    "requires_positive_y": any,
    "_skip_test": any,
    "_xfail_checks": dictmerge,
    "stateless": all,
    "X_types": intersection,
}


def tag_accumulate(model_tags: Iterable[Mapping[str, Any]]) -> Dict[str, Any]:
    """Accumulate all model_tags so that the accumulated tags suppports each model.

    This will use the defaults too from sklearn if it doesn't exist.

    Parameters
    ----------
    model_tags : Iterable[Mapping[str, Any]]
        An iterable over the tags for each model

    Returns
    -------
    Dict[str, Any]
        The tags which allow all models to be supported
    """
    accumulated_tags: Dict[str, List[Any]] = defaultdict(list)

    for tags in model_tags:
        for k, v in tags.items():
            accumulated_tags[k].append(v)

    results: Dict[str, Any] = {}
    for k, v in accumulated_tags.items():
        f = accumulation_method[k]
        results[k] = f(v)

    results = {**_DEFAULT_TAGS, **results}

    return results
