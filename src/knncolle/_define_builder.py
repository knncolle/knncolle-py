from functools import singledispatch
from typing import Tuple

from ._classes import Parameters


@singledispatch
def define_builder(param: Parameters) -> Tuple:
    """
    Create a :py:class:`~knncolle.Builder` instance for a given nearest neighbor search algorithm.
    The ``Builder`` is used in :py:func:`~knncolle.build_index` to create a search index from a matrix of observations.

    Args:
        param:
            Parameters for a particular search algorithm.

    Returns:
        Tuple where the first element is a :py:class:`~knncolle.Builder` and the second element is an instance of the relevant :py:class:`~knncolle.GenericIndex` subclass.

    Raises:
        NotImplementedError: if no method was implemented for the specified algorithm.

    Examples:
        >>> import knncolle
        >>> knncolle.define_builder(knncolle.KmknnParameters())
    """
    raise NotImplementedError("no available method for '" + str(type(param)) + "'")
