// chemprop_accel native CPU kernels.
//
// Registered as torch custom operators in the ``chemprop_accel`` namespace. No Python.h
// dependency: the library is loaded with ``torch.ops.load_library``.
//
// Operators (all CPU, float32/float64, int64 indices):
//   edge_message(H[E,d], dst[E], src[E], rev[E]|empty, n)  -> M[E,d],  M[e] = sum_{k: dst[k]=src[e]} H[k] - H[rev[e]]
//   edge_message_backward(g[E,d], dst, src, rev|empty, n)  -> dH[E,d]
//
// ``variant`` selects an algorithm for the atom-level reduction (see docs/optimisations.md):
//   0 = column-tiled scatter (no atomics, no adjacency construction)
//   1 = CSR build + atom-parallel gather-reduce
//   2 = direct per-edge neighbour sum from CSR (no atom buffer)  [default; fastest in benchmarks]
#include <ATen/ATen.h>
#include <ATen/Parallel.h>
#include <torch/library.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace {

constexpr int64_t kSerialElems = 1 << 14;  // below this many touched elements run serially
constexpr int64_t kGrainElems = 1 << 14;   // target elements per parallel chunk

#if defined(__GNUC__) && !defined(__clang__) && defined(__x86_64__) && defined(__linux__) && \
    !defined(CHEMPROP_ACCEL_NO_CLONES)
// Runtime ISA dispatch (AVX-512 / AVX2+FMA / baseline) through GCC function multi-versioning.
#define CA_CLONES __attribute__((target_clones("avx512f", "avx2", "default")))
#else
#define CA_CLONES
#endif

// ---- row primitives (compiled once per ISA clone) -------------------------------------------
#define CA_ROW_OPS(T, SUF)                                                                      \
  CA_CLONES void acc_##SUF(T* __restrict a, const T* __restrict b, int64_t n) {                 \
    for (int64_t i = 0; i < n; ++i) a[i] += b[i];                                               \
  }                                                                                             \
  CA_CLONES void sub_##SUF(T* __restrict o, const T* __restrict a, const T* __restrict b,       \
                           int64_t n) {                                                         \
    for (int64_t i = 0; i < n; ++i) o[i] = a[i] - b[i];                                         \
  }                                                                                             \
  CA_CLONES void sub_acc_##SUF(T* __restrict a, const T* __restrict b, int64_t n) {             \
    for (int64_t i = 0; i < n; ++i) a[i] -= b[i];                                               \
  }
CA_ROW_OPS(float, f32)
CA_ROW_OPS(double, f64)
#undef CA_ROW_OPS

inline void acc(float* a, const float* b, int64_t n) { acc_f32(a, b, n); }
inline void acc(double* a, const double* b, int64_t n) { acc_f64(a, b, n); }
inline void sub(float* o, const float* a, const float* b, int64_t n) { sub_f32(o, a, b, n); }
inline void sub(double* o, const double* a, const double* b, int64_t n) { sub_f64(o, a, b, n); }
inline void sub_acc(float* a, const float* b, int64_t n) { sub_acc_f32(a, b, n); }
inline void sub_acc(double* a, const double* b, int64_t n) { sub_acc_f64(a, b, n); }

inline void check_index(const int64_t* idx, int64_t R, int64_t n, const char* what) {
  for (int64_t r = 0; r < R; ++r)
    TORCH_CHECK(idx[r] >= 0 && idx[r] < n, "chemprop_accel: ", what, " index ", idx[r],
                " out of range [0, ", n, ")");
}

// CSR of "which rows map to each bucket": offsets[n+1], order[R] (stable counting sort).
struct Csr {
  std::vector<int64_t> offsets, order;
};
inline Csr build_csr(const int64_t* idx, int64_t R, int64_t n) {
  Csr c;
  c.offsets.assign(n + 1, 0);
  for (int64_t r = 0; r < R; ++r) ++c.offsets[idx[r] + 1];
  for (int64_t i = 0; i < n; ++i) c.offsets[i + 1] += c.offsets[i];
  c.order.resize(R);
  std::vector<int64_t> cur(c.offsets.begin(), c.offsets.end() - 1);
  for (int64_t r = 0; r < R; ++r) c.order[cur[idx[r]]++] = r;
  return c;
}

// ---- kernel 1: A = scatter_sum(X, idx) -------------------------------------------------------
template <typename T>
void scatter_tiled(const T* X, const int64_t* idx, T* A, int64_t R, int64_t d) {
  // Parallelise over column tiles: threads own disjoint columns of A so no atomics are needed.
  const int64_t nthreads = std::max<int64_t>(1, at::get_num_threads());
  int64_t tile = ((d + 2 * nthreads - 1) / (2 * nthreads) + 15) / 16 * 16;
  tile = std::max<int64_t>(16, std::min<int64_t>(tile, 256));
  const int64_t n_tiles = (d + tile - 1) / tile;
  auto body = [&](int64_t t0, int64_t t1) {
    for (int64_t t = t0; t < t1; ++t) {
      const int64_t c0 = t * tile, w = std::min(tile, d - c0);
      for (int64_t r = 0; r < R; ++r) acc(A + idx[r] * d + c0, X + r * d + c0, w);
    }
  };
  if (R * d < kSerialElems || n_tiles == 1) {
    body(0, n_tiles);
  } else {
    at::parallel_for(0, n_tiles, 1, body);
  }
}

template <typename T>
void scatter_csr(const T* X, const Csr& c, T* A, int64_t n, int64_t d) {
  auto body = [&](int64_t i0, int64_t i1) {
    for (int64_t i = i0; i < i1; ++i)
      for (int64_t p = c.offsets[i]; p < c.offsets[i + 1]; ++p)
        acc(A + i * d, X + c.order[p] * d, d);
  };
  if (c.order.size() * d < kSerialElems) {
    body(0, n);
  } else {
    at::parallel_for(0, n, std::max<int64_t>(1, kGrainElems / std::max<int64_t>(d, 1)), body);
  }
}

// ---- kernel 2: out[e] = A[a_idx[e]] - X[b_idx[e]]  (b may be null) ---------------------------
template <typename T>
void gather_sub(const T* A, const int64_t* a_idx, const T* X, const int64_t* b_idx, T* out,
                int64_t R, int64_t d) {
  auto body = [&](int64_t r0, int64_t r1) {
    for (int64_t r = r0; r < r1; ++r) {
      if (b_idx)
        sub(out + r * d, A + a_idx[r] * d, X + b_idx[r] * d, d);
      else
        std::memcpy(out + r * d, A + a_idx[r] * d, d * sizeof(T));
    }
  };
  if (R * d < kSerialElems) {
    body(0, R);
  } else {
    at::parallel_for(0, R, std::max<int64_t>(1, kGrainElems / std::max<int64_t>(d, 1)), body);
  }
}

// direct variant: out[e] = sum_{k in in(src[e])} H[k] - H[rev[e]], no atom buffer
template <typename T>
void direct_edge(const T* H, const Csr& dst_csr, const int64_t* src, const int64_t* rev, T* out,
                 int64_t E, int64_t d) {
  auto body = [&](int64_t e0, int64_t e1) {
    for (int64_t e = e0; e < e1; ++e) {
      T* o = out + e * d;
      std::memset(o, 0, d * sizeof(T));
      const int64_t v = src[e];
      for (int64_t p = dst_csr.offsets[v]; p < dst_csr.offsets[v + 1]; ++p)
        acc(o, H + dst_csr.order[p] * d, d);
      if (rev) sub_acc(o, H + rev[e] * d, d);
    }
  };
  if (E * d < kSerialElems) {
    body(0, E);
  } else {
    at::parallel_for(0, E, std::max<int64_t>(1, kGrainElems / std::max<int64_t>(d, 1)), body);
  }
}

at::Tensor prep(const at::Tensor& t, const char* name, bool is_index) {
  TORCH_CHECK(t.device().is_cpu(), "chemprop_accel: ", name, " must be a CPU tensor");
  if (is_index) {
    TORCH_CHECK(t.scalar_type() == at::kLong && t.dim() == 1, "chemprop_accel: ", name,
                " must be a 1-D int64 tensor");
  } else {
    TORCH_CHECK(t.dim() == 2, "chemprop_accel: ", name, " must be 2-D");
  }
  return t.contiguous();
}

// A = scatter_sum(X, idx, n) using the selected variant
template <typename T>
void scatter_dispatch(const T* X, const int64_t* idx, T* A, int64_t R, int64_t n, int64_t d,
                      int64_t variant) {
  if (variant == 1 && R > 0) {
    Csr c = build_csr(idx, R, n);
    scatter_csr<T>(X, c, A, n, d);
  } else {
    scatter_tiled<T>(X, idx, A, R, d);
  }
}

// core: M[e] = (sum_{k: a_key[k]=b_key[e]} X[k]) - X[rev[e]]
at::Tensor edge_reduce(const at::Tensor& X_, const at::Tensor& group_, const at::Tensor& read_,
                       const at::Tensor& rev_, int64_t n, int64_t variant) {
  auto X = prep(X_, "H", false);
  auto group = prep(group_, "dst", true);
  auto read = prep(read_, "src", true);
  const bool has_rev = rev_.numel() > 0;
  at::Tensor rev = has_rev ? prep(rev_, "rev", true) : at::empty({0}, group.options());
  const int64_t E = X.size(0), d = X.size(1);
  TORCH_CHECK(group.size(0) == E && read.size(0) == E, "chemprop_accel: edge array length mismatch");
  TORCH_CHECK(!has_rev || rev.size(0) == E, "chemprop_accel: rev length mismatch");
  check_index(group.data_ptr<int64_t>(), E, n, "dst");
  check_index(read.data_ptr<int64_t>(), E, n, "src");
  if (has_rev) check_index(rev.data_ptr<int64_t>(), E, E, "rev");
  auto out = at::empty({E, d}, X.options());
  const int64_t* g = group.data_ptr<int64_t>();
  const int64_t* rd = read.data_ptr<int64_t>();
  const int64_t* rv = has_rev ? rev.data_ptr<int64_t>() : nullptr;
  AT_DISPATCH_FLOATING_TYPES(X.scalar_type(), "ca_edge_reduce", [&] {
    const scalar_t* Xp = X.data_ptr<scalar_t>();
    scalar_t* op = out.data_ptr<scalar_t>();
    if (variant == 2 && E > 0) {
      Csr c = build_csr(g, E, n);
      direct_edge<scalar_t>(Xp, c, rd, rv, op, E, d);
    } else {
      auto A = at::zeros({n, d}, X.options());
      scatter_dispatch<scalar_t>(Xp, g, A.data_ptr<scalar_t>(), E, n, d, variant);
      gather_sub<scalar_t>(A.data_ptr<scalar_t>(), rd, Xp, rv, op, E, d);
    }
  });
  return out;
}

at::Tensor edge_message(const at::Tensor& H, const at::Tensor& dst, const at::Tensor& src,
                        const at::Tensor& rev, int64_t n, int64_t variant) {
  return edge_reduce(H, dst, src, rev, n, variant);
}

// Backward. out[e] = A[src[e]] - H[rev[e]],  A[v] = sum_{k:dst[k]=v} H[k].
//   dH[k] = G[dst[k]] - sum_{e: rev[e]=k} g[e],   G[u] = sum_{e: src[e]=u} g[e]
// When rev is an involution (always true for Chemprop graphs) the second term is g[rev[k]],
// so backward is the same fused kernel with the roles of src and dst swapped. Otherwise a
// general (serial-per-tile) scatter is used.
at::Tensor edge_message_backward(const at::Tensor& g_, const at::Tensor& dst_,
                                 const at::Tensor& src_, const at::Tensor& rev_, int64_t n,
                                 int64_t variant) {
  const bool has_rev = rev_.numel() > 0;
  bool involution = true;
  at::Tensor rev = rev_;
  if (has_rev) {
    rev = prep(rev_, "rev", true);
    const int64_t* rv = rev.data_ptr<int64_t>();
    const int64_t E = rev.size(0);
    check_index(rv, E, E, "rev");
    for (int64_t e = 0; e < E && involution; ++e) involution = rv[rv[e]] == e;
  }
  if (!has_rev || involution) {
    return edge_reduce(g_, src_, dst_, rev_, n, variant);
  }
  // general permutation / repeated reverse indices
  auto g = prep(g_, "grad", false);
  auto dst = prep(dst_, "dst", true);
  auto src = prep(src_, "src", true);
  const int64_t E = g.size(0), d = g.size(1);
  auto dH = at::zeros({E, d}, g.options());
  auto G = at::zeros({n, d}, g.options());
  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "ca_edge_bwd_scatter", [&] {
    scatter_dispatch<scalar_t>(g.data_ptr<scalar_t>(), src.data_ptr<int64_t>(), G.data_ptr<scalar_t>(), E, n, d,
                               variant);
  });
  AT_DISPATCH_FLOATING_TYPES(g.scalar_type(), "ca_edge_bwd_general", [&] {
    gather_sub<scalar_t>(G.data_ptr<scalar_t>(), dst.data_ptr<int64_t>(), nullptr, nullptr,
                         dH.data_ptr<scalar_t>(), E, d);
    const scalar_t* gp = g.data_ptr<scalar_t>();
    const int64_t* rv = rev.data_ptr<int64_t>();
    scalar_t* dp = dH.data_ptr<scalar_t>();
    for (int64_t e = 0; e < E; ++e) sub_acc(dp + rv[e] * d, gp + e * d, d);
  });
  return dH;
}

}  // namespace

TORCH_LIBRARY(chemprop_accel, m) {
  m.def("edge_message(Tensor H, Tensor dst, Tensor src, Tensor rev, int n, int variant=2) -> Tensor");
  m.def("edge_message_backward(Tensor g, Tensor dst, Tensor src, Tensor rev, int n, int variant=2) -> Tensor");
}

TORCH_LIBRARY_IMPL(chemprop_accel, CPU, m) {
  m.impl("edge_message", &edge_message);
  m.impl("edge_message_backward", &edge_message_backward);
}
