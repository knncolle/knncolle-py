import sys

if sys.version_info[:2] >= (3, 8):
    # TODO: Import directly (no need for conditional) when `python_requires = >= 3.8`
    from importlib.metadata import PackageNotFoundError, version  # pragma: no cover
else:
    from importlib_metadata import PackageNotFoundError, version  # pragma: no cover

try:
    # Change here if project is renamed and does not equal the package name
    _dist_name = __name__
    __version__ = version(_dist_name)
except PackageNotFoundError:  # pragma: no cover
    __version__ = "unknown"
finally:
    del version, PackageNotFoundError

from ._classes import Parameters, Index, GenericIndex
from ._annoy import AnnoyParameters, AnnoyIndex
from ._build_index import build_index
from ._define_builder import define_builder
from ._exhaustive import ExhaustiveParameters, ExhaustiveIndex
from ._find_distance import find_distance
from ._find_knn import find_knn, FindKnnResults
from ._find_neighbors import find_neighbors, FindNeighborsResults
from ._hnsw import HnswParameters, HnswIndex
from ._kmknn import KmknnParameters, KmknnIndex
from ._query_distance import query_distance
from ._query_knn import query_knn, QueryKnnResults
from ._query_neighbors import query_neighbors, QueryNeighborsResults
from ._vptree import VptreeParameters, VptreeIndex


def includes() -> str:
    """Provides access to ``knncolle_py.h`` C++ header.

    Returns:
        Path to a directory containing the header.
    """
    import os
    import inspect
    dirname = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
    return os.path.join(dirname, "include")


# For proper documentation with sphinx.
__all__ = []
for _name in dir():
    if not _name.startswith("_") and _name != "sys":
        __all__.append(_name)
