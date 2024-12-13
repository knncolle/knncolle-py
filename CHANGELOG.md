# Changelog

## Version 0.1.1

- Cast pointers to/from `uintptr_t` so that downstream packages aren't forced to rely on **pybind10** converters.
- Added a `knncolle_py.h` header to ensure developers use the correct types during casting.

## Version 0.1.0

Bindings to the algorithms in the **knncolle** package.
