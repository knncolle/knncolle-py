#include "knncolle_py.h"
#include "pybind11/pybind11.h"

// Turn off manual vectorization always, to avoid small inconsistencies in
// distance calculations across otherwise-compliant machines. 
#define NO_MANUAL_VECTORIZATION

#include "knncolle_hnsw/knncolle_hnsw.hpp"

uintptr_t create_hnsw_builder(int nlinks, int ef_construct, int ef_search, std::string distance) {
    knncolle_hnsw::HnswOptions<uint32_t, float> opt;
    opt.num_links = nlinks;
    opt.ef_construction = ef_construct;
    opt.ef_search = ef_search;
    auto tmp = std::make_unique<knncolle_py::WrappedBuilder>();

    if (distance == "Manhattan") {
        opt.distance_options.create = [&](int dim) -> hnswlib::SpaceInterface<float>* {
            return new knncolle_hnsw::ManhattanDistance<float>(dim);
        };
        tmp->ptr.reset(new knncolle_hnsw::HnswBuilder<knncolle_py::SimpleMatrix, knncolle_py::Distance>(opt));

    } else if (distance == "Euclidean") {
        tmp->ptr.reset(new knncolle_hnsw::HnswBuilder<knncolle_py::SimpleMatrix, knncolle_py::Distance>(opt));

    } else if (distance == "Cosine") {
        tmp->ptr.reset(
            new knncolle::L2NormalizedBuilder<knncolle_py::SimpleMatrix, knncolle_py::Distance>(
                new knncolle_hnsw::HnswBuilder<
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

void init_hnsw(pybind11::module& m) {
    m.def("create_hnsw_builder", &create_hnsw_builder);
}
