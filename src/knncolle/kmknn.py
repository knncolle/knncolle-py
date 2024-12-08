from .classes import Parameters, GenericIndex
from typing import Literal, Tuple

from . import lib_knncolle as lib


class KmknnParameters(Parameters):
    """Parameters for the k-means k-nearest neighbors (KMKNN) algorithm."""

    def __init__(
        distance: Literal["Euclidean", "Manhattan", "Cosine"] = "Euclidean",
    ):
        """
        Args:
            distance:
                Distance metric for index construction and search. This should
                be one of ``Euclidean``, ``Manhattan`` or ``Cosine``.
        """
        self.distance = distance

    @property
    def distance(self) -> str:
        """
        Return:
            Distance metric, see :meth:`~__init__()`.
        """
        return self._distance

    @distance.setter
    def distance(self, distance: str):
        """
        Args:
            distance:
                Distance metric, see :meth:`~__init__()`.
        """
        if value not in ["Euclidean", "Manhattan", "Cosine"]:
            raise ValueError("unsupported 'distance'")
        self._distance = value


class KmknnIndex(GenericIndex):
    """A prebuilt index for the k-means k-nearest neighbors algorithm, created
    by :py:func:`~knncolle.define_builder.define_builder` with a
    :py:class:`~knncolle.kmknn.KmknnParameters` instance.
    """

    def __init__(ptr):
        """
        Args:
            ptr:
                Shared pointer to a `knncolle::Prebuilt<uint32_t, uint32_t,
                double>`, created and wrapped by pybind11.
        """
        self._ptr = ptr

    @property
    def ptr(self):
        """
        Return:
            See :py:meth:`~__init-_`.
        """
        return self._ptr


@define_builder.register
def _define_builder_kmknn(x: AnnoyParameters) -> Tuple:
    return (lib.create_kmknn_builder(x.distance), KmknnIndex)
