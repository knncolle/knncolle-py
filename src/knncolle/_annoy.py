from typing import Literal, Optional, Tuple

from . import _lib_knncolle as lib
from ._define_builder import define_builder
from ._classes import Index, Builder, GenericIndex, Parameters


class AnnoyParameters(Parameters):
    """
    Parameters for the Approximate Nearest Neighbors Oh Yeah (Annoy) algorithm, see `here <https://github.com/spotify/annoy>`_ for details.
    This can be used in :py:func:`~knncolle.build_index` or :py:func:`~knncolle.define_builder`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.AnnoyParameters()
        >>> params.distance
        >>> params.search_mult
    """

    def __init__(
        self,
        num_trees: int = 50, 
        search_mult: Optional[float] = None,
        distance: Literal["Euclidean", "Manhattan", "Cosine"] = "Euclidean",
    ):
        """
        Args:
            num_trees:
                Number of trees to use to generate the search index.
                More trees increase accuracy at the cost of more computational work, in terms of both the indexing time and search speed.

            search_mult:
                Multiplier for the number of observations to search.
                Specifically, the product of ``k`` and ``search_mult`` is used to define the number of points to search exhaustively and dictates the balance between search speed and accuracy.
                If ``None``, this defaults to the value of ``num_trees``.

            distance:
                Distance metric for index construction and search.
        """
        self.num_trees = num_trees
        self.search_mult = search_mult
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

    @property
    def num_trees(self) -> int:
        """Number of trees, see :meth:`~__init__()`."""
        return self._num_trees

    @num_trees.setter
    def num_trees(self, num_trees: int):
        """
        Args:
            num_trees:
                Number of trees, see :meth:`~__init__()`.
        """
        if num_trees < 1:
            raise ValueError("'num_trees' should be a positive integer")
        self._num_trees = num_trees

    @property
    def search_mult(self) -> int:
        """Search multiplier, see :meth:`~__init__()`."""
        return self._search_mult

    @search_mult.setter
    def search_mult(self, search_mult: Optional[float]):
        """
        Args:
            search_mult:
                Search multiplier, see :meth:`~__init__()`.
        """
        if search_mult is None:
            search_mult = float(self._num_trees)
        if search_mult <= 1:
            raise ValueError("'search_mult' should be greater than 1")
        self._search_mult = search_mult


class AnnoyIndex(GenericIndex):
    """
    Prebuilt index for the Approximate Nearest Neighbors Oh Yeah (Annoy) algorithm.
    This is typically created by :py:func:`~knncolle.build_index` with an :py:class:`~AnnoyParameters` object,
    and can be used in functions like :py:func:`~knncolle.find_knn`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.AnnoyParameters()
        >>> import numpy
        >>> y = numpy.random.rand(200, 10)
        >>> idx = knncolle.build_index(params, y)
        >>> type(idx)
    """

    def __init__(self, ptr: int):
        """
        Args:
            ptr:
                Address of a ``knncolle_py::WrappedPrebuilt`` containing an Annoy search index, allocated in C++.
        """
        super().__init__(ptr)


@define_builder.register
def _define_builder_annoy(x: AnnoyParameters) -> Tuple:
    return (Builder(lib.create_annoy_builder(x.num_trees, x.search_mult, x.distance)), AnnoyIndex)
