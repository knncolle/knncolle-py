from typing import Literal, Tuple

from . import _lib_knncolle as lib
from ._classes import Parameters, GenericIndex, Builder
from ._define_builder import define_builder


class KmknnParameters(Parameters):
    """
    Parameters for the k-means k-nearest neighbors (KMKNN) algorithm.
    This can be used in :py:func:`~knncolle.build_index` or :py:func:`~knncolle.define_builder`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.KmknnParameters()
        >>> params.distance
    """

    def __init__(
        self,
        distance: Literal["Euclidean", "Manhattan", "Cosine"] = "Euclidean",
    ):
        """
        Args:
            distance:
                Distance metric for index construction and search.
        """
        self.distance = distance

    @property
    def distance(self) -> str:
        """Distance metric, see :meth:`~__init__()`."""
        return self._distance

    @distance.setter
    def distance(self, distance: str):
        """
        Args:
            distance:
                Distance metric, see :meth:`~__init__()`.
        """
        if distance not in ["Euclidean", "Manhattan", "Cosine"]:
            raise ValueError("unsupported 'distance'")
        self._distance = distance 


class KmknnIndex(GenericIndex):
    """
    Prebuilt index for the k-means k-nearest neighbors algorithm.
    This is typically created by :py:func:`~knncolle.build_index` with an :py:class:`~KmknnParameters` object,
    and can be used in functions like :py:func:`~knncolle.find_knn`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.KmknnParameters()
        >>> import numpy
        >>> y = numpy.random.rand(200, 10)
        >>> idx = knncolle.build_index(params, y)
        >>> type(idx)
    """

    def __init__(self, ptr):
        """
        Args:
            ptr:
                Address of a ``knncolle_py::WrappedPrebuilt`` containing a KMKNN search index, allocated in C++.
        """
        super().__init__(ptr)


@define_builder.register
def _define_builder_kmknn(x: KmknnParameters) -> Tuple:
    return (Builder(lib.create_kmknn_builder(x.distance)), KmknnIndex)
