from typing import Dict, Iterable, List, Set, TypeVar

from functools import reduce
from itertools import chain

T = TypeVar("T")


def intersection(items: Iterable[Iterable[T]]) -> Set[T]:
    """Does an intersection over all collection of items

    Parameters
    ----------
    items : Iterable[Iterable[T]]
        A list of lists


    Returns
    -------
    Set[T]
        The intersection of all items
    """
    return set(reduce(lambda s1, s2: set(s1) & set(s2), items))


def union(items: Iterable[Iterable[T]]) -> Set[T]:
    """Does a union over all collection of items

    Parameters
    ----------
    items : Iterable[Iterable[T]]
        A list of lists

    Returns
    -------
    Set[T]
        The intersection of all items
    """
    return set(chain.from_iterable(items))


def dictmerge(items: Iterable[Dict], recursive: bool = False) -> Dict:
    """Does a dictionary merge

    Parameters
    ----------
    items : Iterable[Dict]
        An iterable of dicts to merge

    Returns
    -------
    Dict
        The merged dictionary
    """
    if recursive is True:
        raise NotImplementedError("Don't have recursive yet")

    results = {}
    for d in items:
        if not isinstance(d, Dict):
            continue
        for k, v in d.items():
            if k in results:
                if isinstance(results[k], List):
                    results[k].append(v)
                else:
                    results[k] = [results[k], v]
            else:
                results[k] = v

    return results
