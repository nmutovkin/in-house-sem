#pragma once

#include <cstdint>

#include "insem/core/build_config.hpp"

// Marks small point-wise functions that are shared between the CPU and CUDA
// backends (see docs/DESIGN.md §7.3).
#if defined(__CUDACC__)
#define INSEM_HD __host__ __device__
#else
#define INSEM_HD
#endif

namespace insem {

// Floating point type of solver fields. Mesh, basis and geometry setup is
// always done in double precision and converted when blocks are built.
#if INSEM_REAL_IS_DOUBLE
using real_t = double;
#else
using real_t = float;
#endif

// Index of elements and points on one rank.
using index_t = std::int32_t;

inline constexpr const char* version() { return INSEM_VERSION_STRING; }

}  // namespace insem
