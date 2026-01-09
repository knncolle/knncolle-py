from typing import Literal, Optional, Tuple

from . import _lib_knncolle as lib
from ._classes import Parameters, GenericIndex, Builder
from ._define_builder import define_builder


class HnswParameters(Parameters):
    """
    Parameters for the hierarchical navigable small worlds (HNSW) algorithm, see `here <https://github.com/nmslib/hnswlib>`_ for details.
    This can be used in :py:func:`~knncolle.build_index` or :py:func:`~knncolle.define_builder`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.HnswParameters()
        >>> params.distance
        >>> params.ef_search
    """

    def __init__(
        self,
        num_links: int = 16, 
        ef_construction: int = 200,
        ef_search: int = 10,
        distance: Literal["Euclidean", "Manhattan", "Cosine"] = "Euclidean",
    ):
        """
        Args:
            num_links:
                Number of bi-directional links to create per observation during index construction.
                Larger values improve accuracy at the expense of speed and memory usage.

            ef_construction:
                Size of the dynamic list for index generation.
                Larger values improve the quality of the index at the expense of time.

            ef_search
                Size of the dynamic list for neighbor searching.
                Larger values improve accuracy at the expense of a slower search.

            distance:
                Distance metric for index construction and search.
        """
        self.num_links = num_links
        self.ef_construction = ef_construction
        self.ef_search = ef_search
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
    def num_links(self) -> int:
        """Number of links, see :meth:`~__init__()`."""
        return self._num_links

    @num_links.setter
    def num_links(self, num_links: int):
        """
        Args:
            num_links:
                Number of links, see :meth:`~__init__()`.
        """
        if num_links < 1:
            raise ValueError("'num_links' should be a positive integer")
        self._num_links = num_links

    @property
    def ef_construction(self) -> int:
        """Size of the dynamic list during index construction, see :meth:`~__init__()`."""
        return self._ef_construction

    @ef_construction.setter
    def ef_construction(self, ef_construction: int):
        """
        Args:
            ef_construction:
                Size of the dynamic list during index construction, see :meth:`~__init__()`.
        """
        if ef_construction < 1:
            raise ValueError("'ef_construction' should be a positive integer")
        self._ef_construction = ef_construction

    @property
    def ef_search(self) -> int:
        """Size of the dynamic list during search, see :meth:`~__init__()`."""
        return self._ef_search

    @ef_search.setter
    def ef_search(self, ef_search: int):
        """
        Args:
            ef_search:
                Size of the dynamic list during search, see :meth:`~__init__()`.
        """
        if ef_search < 1:
            raise ValueError("'ef_search' should be a positive integer")
        self._ef_search = ef_search


class HnswIndex(GenericIndex):
    """
    Prebuilt index for the hierarchical navigable small worlds (HNSW) algorithm,
    This is typically created by :py:func:`~knncolle.build_index` with an :py:class:`~HnswParameters` object,
    and can be used in functions like :py:func:`~knncolle.find_knn`.

    Examples:
        >>> import knncolle
        >>> params = knncolle.HnswParameters()
        >>> import numpy
        >>> y = numpy.random.rand(200, 10)
        >>> idx = knncolle.build_index(params, y)
        >>> type(idx)
    """

    def __init__(self, ptr):
        """
        Args:
            ptr:
                Address of a ``knncolle_py::WrappedPrebuilt`` containing a HNSW search index, allocated in C++.
        """
        super().__init__(ptr)


@define_builder.register
def _define_builder_hnsw(x: HnswParameters) -> Tuple:
    return (Builder(lib.create_hnsw_builder(x.num_links, x.ef_construction, x.ef_search, x.distance)), HnswIndex)
