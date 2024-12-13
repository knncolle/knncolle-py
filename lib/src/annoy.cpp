#include "knncolle_py.h"
#include "pybind11/pybind11.h"

#include <memory>
#include <stdexcept>

// Turn off manual vectorization always, to avoid small inconsistencies in
// distance calculations across otherwise-compliant machines. 
#define NO_MANUAL_VECTORIZATION 1

#include "knncolle_annoy/knncolle_annoy.hpp"

uintptr_t create_annoy_builder(int num_trees, double search_mult, std::string distance) {
    knncolle_annoy::AnnoyOptions opt;
    opt.num_trees = num_trees;
    opt.search_mult = search_mult;
    auto tmp = std::make_unique<knncolle_py::WrappedBuilder>();

    if (distance == "Manhattan") {
        tmp->ptr.reset(new knncolle_annoy::AnnoyBuilder<Annoy::Manhattan, knncolle_py::SimpleMatrix, knncolle_py::Distance>(opt));

    } else if (distance == "Euclidean") {
        tmp->ptr.reset(new knncolle_annoy::AnnoyBuilder<Annoy::Euclidean, knncolle_py::SimpleMatrix, knncolle_py::Distance>(opt));

    } else if (distance == "Cosine") {
        tmp->ptr.reset(
            new knncolle::L2NormalizedBuilder<knncolle_py::SimpleMatrix, knncolle_py::Distance>(
                new knncolle_annoy::AnnoyBuilder<
                    Annoy::Euclidean,
                    knncolle::L2NormalizedMatrix<knncolle_py::SimpleMatrix>,
                    double
                >(opt)
            )
        );

    } else {
        throw std::runtime_error("unknown distance type '" + distance + "'");
    }

    return reinterpret_cast<uintptr_t>(static_cast<void*>(tmp.release()));
}

void init_annoy(pybind11::module& m) {
    m.def("create_annoy_builder", &create_annoy_builder);
}
