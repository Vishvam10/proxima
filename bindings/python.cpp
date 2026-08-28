#include "proxima/hnsw.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <nanobind/nanobind.h>
#include <nanobind/ndarray.h>
#include <nanobind/stl/vector.h>
#include <stdexcept>
#include <vector>

namespace nb = nanobind;

using namespace nb::literals;

namespace {

std::vector<std::vector<float>>
ndarray_to_vectors(nb::ndarray<const float, nb::c_contig> data) {
    if (data.ndim() != 2) {
        throw std::invalid_argument(
            "data must be a 2D float32 array"
        );
    }

    const std::size_t n = data.shape(0);
    const std::size_t dim = data.shape(1);

    std::vector<std::vector<float>> result(
        n,
        std::vector<float>(dim)
    );

    const float* ptr = data.data();

    for (std::size_t i = 0; i < n; ++i) {
        std::copy(
            ptr + i * dim,
            ptr + (i + 1) * dim,
            result[i].begin()
        );
    }

    return result;
}

} // namespace

NB_MODULE(_proxima, m) {
    nb::enum_<proxima::DistanceType>(m, "DistanceType")
        .value("L2", proxima::DistanceType::L2)
        .value(
            "INNER_PRODUCT",
            proxima::DistanceType::INNER_PRODUCT
        )
        .value(
            "COSINE",
            proxima::DistanceType::COSINE
        );

    nb::class_<proxima::HnswCPU>(m, "HnswCPU")
        .def(
            nb::init<
                int,
                int,
                std::uint32_t,
                proxima::DistanceType,
                bool
            >(),
            "M"_a = 16,
            "ef_construction"_a = 200,
            "seed"_a = 42,
            "distance_type"_a = proxima::DistanceType::L2,
            "force_scalar"_a = false
        )

        .def(
            "__len__",
            &proxima::HnswCPU::size
        )

        .def(
            "create",
            [](proxima::HnswCPU& self,
               nb::ndarray<const float, nb::c_contig> data) {
                self.create(ndarray_to_vectors(data));
            }
        )

        .def(
            "add",
            [](proxima::HnswCPU& self,
               nb::ndarray<const float, nb::c_contig> embedding) {
                if (embedding.ndim() != 1) {
                    throw std::invalid_argument(
                        "embedding must be a 1D float32 array"
                    );
                }

                std::vector<float> values(
                    embedding.data(),
                    embedding.data() + embedding.shape(0)
                );

                self.add(values);
            }
        )

        .def(
            "search",
            [](proxima::HnswCPU& self,
               nb::ndarray<const float, nb::c_contig> query,
               int k,
               int ef_search) {
                if (query.ndim() != 1) {
                    throw std::invalid_argument(
                        "query must be a 1D float32 array"
                    );
                }

                std::vector<float> values(
                    query.data(),
                    query.data() + query.shape(0)
                );

                return self.search(
                    values,
                    k,
                    ef_search
                );
            },
            "query"_a,
            "k"_a,
            "ef_search"_a = 50
        )

        .def(
            "size",
            &proxima::HnswCPU::size
        )

        .def(
            "dimension",
            &proxima::HnswCPU::dimension
        )

        .def(
            "M",
            &proxima::HnswCPU::M
        )

        .def(
            "ef_construction",
            &proxima::HnswCPU::efConstruction
        )

        .def(
            "distance_type",
            &proxima::HnswCPU::distanceType
        );
}