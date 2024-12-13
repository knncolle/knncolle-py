#include "knncolle_py.h"
#include "pybind11/pybind11.h"

#include <memory>
#include <stdexcept>

uintptr_t create_vptree_builder(std::string distance) {
    auto tmp = std::make_unique<knncolle_py::WrappedBuilder>();

    if (distance == "Manhattan") {
        tmp->ptr.reset(new knncolle::VptreeBuilder<knncolle::ManhattanDistance, knncolle_py::SimpleMatrix, knncolle_py::Distance>);

    } else if (distance == "Euclidean") {
        tmp->ptr.reset(new knncolle::VptreeBuilder<knncolle::EuclideanDistance, knncolle_py::SimpleMatrix, knncolle_py::Distance>);

    } else if (distance == "Cosine") {
        tmp->ptr.reset(
            new knncolle::L2NormalizedBuilder(
                new knncolle::VptreeBuilder<
                    knncolle::EuclideanDistance,
                    knncolle::L2NormalizedMatrix<knncolle_py::SimpleMatrix>,
                    double
                >
            )
        );

    } else {
        throw std::runtime_error("unknown distance type '" + distance + "'");
    }

    return reinterpret_cast<uintptr_t>(static_cast<void*>(tmp.release()));
}

void init_vptree(pybind11::module& m) {
    m.def("create_vptree_builder", &create_vptree_builder);
}
