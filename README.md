<!-- These are examples of badges you might want to add to your README:
     please update the URLs accordingly

[![Built Status](https://api.cirrus-ci.com/github/<USER>/knncolle.svg?branch=main)](https://cirrus-ci.com/github/<USER>/knncolle)
[![ReadTheDocs](https://readthedocs.org/projects/knncolle/badge/?version=latest)](https://knncolle.readthedocs.io/en/stable/)
[![Coveralls](https://img.shields.io/coveralls/github/<USER>/knncolle/main.svg)](https://coveralls.io/r/<USER>/knncolle)
[![Conda-Forge](https://img.shields.io/conda/vn/conda-forge/knncolle.svg)](https://anaconda.org/conda-forge/knncolle)
[![Twitter](https://img.shields.io/twitter/url/http/shields.io.svg?style=social&label=Twitter)](https://twitter.com/knncolle)
-->

[![PyPI-Server](https://img.shields.io/pypi/v/knncolle.svg)](https://pypi.org/project/knncolle/)
[![Monthly Downloads](https://static.pepy.tech/badge/knncolle/month)](https://pepy.tech/project/knncolle)
![Unit tests](https://github.com/knncolle/knncolle/actions/workflows/pypi-test.yml/badge.svg)

# Python bindings to knncolle

The **knncolle** Python package implements Python bindings to the [C++ library of the same name](https://github.com/knncolle) for nearest neighbor (NN) searches.
Downstream packages can re-use the NN search algorithms in **knncolle**, either via Python or by directly calling C++ through shared pointers.
This is inspired by the [**BiocNeighbors** Bioconductor package](https://bioconductor/packages/BiocNeighbors), which does the same thing for R packages.
