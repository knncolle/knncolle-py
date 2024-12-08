from functools import singledispatch
from typing import Sequence, Optional, Union
from dataclasses import dataclass

from .classes import Index, GenericIndex
from . import lib_knncolle as lib
from ._utils import process_num_neighbors, process_subset


@dataclass
class FindNeighborsResults:
    """Results of :py:func:`~knncolle.find_neighbors.find_neighbors`.

    ``index`` and ``distance`` are lists where each element is a NumPy array
    that corresponds to an observation in ``X``. Each array contains the
    indices of (for ``index``) or distances to (for ``distance``) the neighbors
    of the corresponding observation within the specified threshold distance.
    For each observation, neighbors are guaranteed to be sorted in order of
    increasing distance. Each element of ``index`` is guaranteed to not contain
    the index of the corresponding observation.

    If ``get_index = False``, ``index`` is set to None.

    If ``get_distance = False``, ``distance`` is set to None.

    If ``subset`` is provided, the length of ``index`` and ``distance`` is
    instead equal to the length of the subset. Each row or list entry
    corresponds to one of the observations in the subset.
    """
    index: Optional[list]
    distance: Optional[list]


@singledispatch
def find_neighbors(
    X: Index,
    threshold: Union[float, Sequence],
    num_threads: int = 1,
    subset: Optional[Sequence] = None, 
    get_index = True,
    get_distance = True,
    **kwargs
) -> numpy.ndarray:
    """Find all neighbors within a certain distance for each observation.

    Args:
        X:
            A prebuilt search index.

        threshold:
            Distance threshold at which to identify neighbors for each
            observation in ``X``. 

            Alternatively, this may be a sequence of non-negative floats of
            length equal to the number of observations in ``X``, specifying
            the distance threshold to search for each observation.

            If ``subset`` is supplied and ``threshold`` is a sequence, it
            should have length equal to ``subset`` instead, and should specify
            the distance threshold for each observation in the subset.

        num_threads:
            Number of threads to use for the search.

        subset:
            Sequence of integers containing the indices of the observations for
            which to identify neighbors. All indices should be non-negative
            and less than the total number of observations.

        get_index:
            Whether to report the indices of each nearest neighbor.

        get_distance:
            Whether to report the distances to each nearest neighbor.

        kwargs:
            Additional arguments to pass to specific methods.

    Return:
        Results of the neighbor search.
    """
    raise NotImplementedError("no available method for '" + type(X) + "'")


@find_neighbors.register
def _find_neighbors_generic(
    X: GenericIndex,
    threshold: Union[int, Sequence],
    num_threads: int = 1,
    subset: Optional[Sequence] = None,
    get_index = True,
    get_distance = True,
    **kwargs
) -> FindNeighborsResults:
    idx, dist = lib.generic_find_neighbors_all(
        X.ptr, 
        threshold = threshold,
        chosen = process_subset(subset), 
        num_threads = num_threads, 
        report_index = get_index,
        report_distance = get_distance
    )
    return FindNeighborsResults(index = idx, distance = dist)
