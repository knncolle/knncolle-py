from functools import singledispatch
from typing import Tuple

from .classes import Parameters


@singledispatch
def define_builder(param: Parameters) -> Tuple:
    """
    Create a builder instance for a given nearest neighbor search algorithm.
    The builder can be used in :py:func:`~knncolle.build_index.build_index`
    to create a search index from a matrix of observations.

    Args:
        param:
            Parameters for a particular search algorithm.

    Return:
        Tuple where the first element is a shared pointer to a
        `knncolle::Builder<knncolle::SimpleMatrix<uint32_t, uint32_t, double>,
        double>` instance, and the second element is a
        :py:class:`~knncolle.classes.GenericIndex` subclass.
    """
    raise NotImplementedError("no available method for '" + type(x) + "'")
