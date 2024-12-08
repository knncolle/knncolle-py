from functools import singledispatch
from typing import Sequence, Optional, Union

from .classes import Index, GenericIndex
from . import lib_knncolle as lib
from ._utils import process_num_neighbors, process_subset


@singledispatch
def query_distance(
    X: Index,
    query: numpy.ndarray,
    num_neighbors: Union[int, Sequence],
    num_threads: int = 1,
    **kwargs
) -> numpy.ndarray:
    """Find the distance to the k-th nearest neighbor in the search index for
    each observation in the query dataset.

    Args:
        X:
            A prebuilt search index.

        query:
            Matrix of coordinates for the query observations. This should be a
            two-dimensional double-precision NumPy array in Fortran order where
            the rows are dimensions and columns are observations. The number of
            dimensions should be consistent with that in ``X``.

        num_neighbors:
            Number of nearest neighbors in ``X`` at which to compute the
            distance from each observation in ``query``, i.e., k. This is
            automatically capped at the total number of observations in ``X``.

            Alternatively, this may be a sequence of integers of length equal
            to the number of observations in ``query``, specifying the neighbor
            at which to compute the distance for each observation.

        num_threads:
            Number of threads to use for the search.

        kwargs:
            Additional arguments to pass to specific methods.

    Return:
        A NumPy array of length equal to the number of observations in
        ``query`` containing the distance to the ``num_neighbor``-th point in
        ``X`` for each observation.
    """
    raise NotImplementedError("no available method for '" + type(X) + "'")


@query_distance.register
def _query_distance_generic(
    X: GenericIndex,
    query: numpy.ndarray,
    num_neighbors: Union[int, Sequence],
    num_threads: int = 1,
    **kwargs
) -> numpy.ndarray:
    num_neighbors, force_variable = process_num_neighbors(num_neighbors)
    return lib.generic_find_knn(
        X.ptr, 
        query = query,
        num_neighbors = num_neighbors,
        force_variable_neighbors = force_variable,
        num_threads = num_threads, 
        last_distance_only = True,
        report_index = False,
        report_distance = False
    )
