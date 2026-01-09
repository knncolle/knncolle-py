from abc import ABC
from . import _lib_knncolle as lib


class Parameters(ABC):
    """
    Abstract base class for the parameters of a nearest neighbor search.
    Each search algorithm should implement a subclass that contains the relevant parameters for controlling index construction or search.
    An instance of a ``Parameters`` subclass can be passed to :py:func:`~knncolle.build_index` to create an instance of corresponding :py:class:`~Index` subclass,
    or to :py:func:`~knncolle.define_builder` to create a :py:class:`~Builder`.
    """
    pass


class Builder:
    """
    Pointer to a search index builder, i.e., ``knncolle_py::WrappedBuilder``, typically created by :py:func:`~knncolle.define_builder`.
    If implemented for an algorithm, it will be called by :py:func:`~knncolle.build_index` to create an instance of corresponding :py:class:`~knncolle.GenericIndex` subclass.
    This pointer can also be passed into package C++ code to build a new neighbor search index via the **knncolle** C++  library.
    The associated memory is automatically freed upon garbage collection.

    Examples:
        >>> import knncolle
        >>> builder = knncolle.define_builder(knncolle.KmknnParameters())
        >>> builder[0].ptr # pass this into C++ code as a std::uintptr_t.
    """

    def __init__(self, ptr: int):
        """
        Args:
            ptr:
                Address of a ``knncolle_py::WrappedBuilder``.
        """
        self._ptr = ptr

    def __del__(self):
        """Frees the builder in C++."""
        lib.free_builder(self._ptr)

    @property
    def ptr(self) -> int:
        """Address of a ``knncolle_py::WrappedBuilder``, to be passed into C++ as a ``std::uintptr_t``; see ``knncolle_py.h`` for details."""
        return self._ptr


class Index(ABC):
    """
    Abstract base class for a prebuilt nearest neighbor-search index.
    This is typically created by :py:class:`~knncolle.build_index` and can be used in functions like :py:func:`~knncolle.find_knn`.
    Each search algorithm should implement its own subclass.
    """
    pass


class GenericIndex(Index):
    """
    Abstract base class for a prebuilt nearest neighbor-search index that holds an address to a ``knncolle_py::WrappedPrebuilt`` instance in C++.
    This pointer can be passed into package C++ code to execute nearest neighbor searches via the **knncolle** C++ library.
    The associated memory is automatically freed upon garbage collection.

    Examples:
        >>> import knncolle
        >>> import numpy
        >>> y = numpy.random.rand(200, 10)
        >>> idx = knncolle.build_index(knncolle.KmknnParameters(), y)
        >>> idx.ptr # pass this into C++ code as a std::uintptr_t.
        >>> idx.num_observations()
        >>> idx.num_dimensions()
    """

    def __init__(self, ptr: int):
        """
        Args:
            ptr:
                Address of a ``knncolle_py::WrappedPrebuilt``.
        """
        self._ptr = ptr

    @property
    def ptr(self) -> int:
        """Address of a ``knncolle_py::WrappedPrebuilt``, to be passed into C++ as a ``uintptr_t``; see ``knncolle_py.h`` for details."""
        return self._ptr

    def __del__(self):
        """Frees the index in C++."""
        lib.free_prebuilt(self._ptr)

    def num_observations(self) -> int:
        """
        Returns:
            Number of observations in this index.
        """
        return lib.generic_num_obs(self._ptr)

    def num_dimensions(self) -> int:
        """
        Returns:
            Number of dimensions in this index.
        """
        return lib.generic_num_dims(self._ptr)
