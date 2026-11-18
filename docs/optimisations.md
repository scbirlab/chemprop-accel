# How it works: what was profiled, fused, and rejected

This page records the engineering decisions, including the things that did **not** help. All numbers are from the
development machine described in {doc}`benchmarks/index` (a shared 2-vCPU Intel Xeon @ 2.3 GHz VM), so they show
*what dominates and what helps*, not what you will get on your hardware.

## 1. Where the time goes in Chemprop

Profiling an eager Chemprop training step (`torch.profiler`, bond MPNN, batch 64, d_h=300, depth 3, 2 threads) shows
the graph plumbing dominates, not the matrix multiplications:

*accel=off · bond · B=64 · d_h=300 · threads=2 · total self-CPU 73.2 ms/step (profiler overhead included)*

| Operator | Self CPU (ms/step) | Share |
| --- | ---: | ---: |
| `aten::mm` | 17.06 | 23% |
| `aten::_index_put_impl_` | 9.71 | 13% |
| `aten::scatter` | 6.69 | 9% |
| `aten::scatter_` | 6.34 | 9% |
| `aten::scatter_reduce_` | 5.43 | 7% |
| `aten::gather` | 4.94 | 7% |
| `aten::copy_` | 4.00 | 5% |
| `aten::fill_` | 1.58 | 2% |
| `aten::clamp_min` | 1.49 | 2% |

The directed-bond message step is

```python
index = bmg.edge_index[1].unsqueeze(1).repeat(1, H.shape[1])  # E x d int64 index tensor
M_all = torch.zeros(V, d).scatter_reduce_(0, index, H, "sum", include_self=False)[bmg.edge_index[0]]
M_rev = H[bmg.rev_edge_index]
return M_all - M_rev
```

which allocates an `E × d` int64 index tensor (8 bytes per float32 of work), zero-fills an atom buffer, scatters,
gathers twice and subtracts, in six PyTorch dispatches; backward repeats the pattern (`scatter`, `gather`,
`index_put`). The same pattern appears in the readout and in the mean/sum aggregations.

## 2. The fused operation

For directed edge `e` (source atom `src[e]`, reverse edge `rev[e]`), Chemprop's message is

$$
M_e \;=\; \sum_{k:\; \mathrm{dst}[k]=\mathrm{src}[e]} H_k \;-\; H_{\mathrm{rev}[e]} .
$$

`chemprop_accel::edge_message` computes it directly into the output with no index expansion and no intermediate
`E × d` tensors. Its analytic gradient is

$$
\frac{\partial L}{\partial H_k} \;=\; G_{\mathrm{dst}[k]} \;-\; \sum_{e:\,\mathrm{rev}[e]=k} g_e,
\qquad G_u=\sum_{e:\,\mathrm{src}[e]=u} g_e ,
$$

and because Chemprop's `rev` is an involution (`rev[rev[e]] = e`) the second term is just `g[rev[k]]`, so the
backward pass is *the same fused kernel with `src` and `dst` swapped*, a property the kernel checks at run time
(O(E) integer comparisons). For arbitrary `rev` (repeated indices) a general scatter path is used. Both are
`gradcheck`-tested in double precision, including isolated atoms, disconnected graphs and empty graphs.

### Algorithm variants (all implemented, benchmarked, one kept as default)

| Variant | Strategy | Atomics | Extra work |
| --- | --- | --- | --- |
| 0 | column-tiled scatter into an atom buffer, then fused gather–subtract | none: threads own disjoint column tiles | zero-fill atom buffer |
| 1 | build CSR of incoming edges, atom-parallel reduction into atom buffer, then fused gather–subtract | none | CSR build |
| **2 (default)** | CSR of incoming edges; each output row is summed directly from the CSR range of its source atom, minus the reverse row | none | CSR build; neighbour rows are re-read (degree is ≤ 4 for organic molecules) |

Inner loops are plain `a += b` / `o = a - b` row operations compiled with GCC function multi-versioning
(`target_clones("avx512f","avx2","default")`) on x86-64 Linux, so the right ISA is chosen at *run time* and a baseline
build runs everywhere; no AVX-512 or AMX is required. Work below ~16k elements runs serially and larger work is split
with `at::parallel_for`, i.e. the existing PyTorch/OpenMP thread pool; the package never creates its own threads, which
avoids oversubscription.

### Measured: op level

Interleaved repeats, medians of 80, µs per call, forward + backward, `d_h=300`. "Pure" is the package's own pure-PyTorch
fallback (`index_add_` + `index_select`, no expanded index).

| Threads | Batch | d_h | Directed edges | Original | Pure fallback | Native v0 | Native v1 | **Native v2 (default)** | v2 vs original | v2 vs fallback |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 8 | 300 | 198 | 978 | 385 | 244 | 182 | **163** | 6.0× | 2.4× |
| 1 | 64 | 300 | 1,960 | 10,114 | 3,822 | 2,269 | 1,858 | **1,525** | 6.6× | 2.5× |
| 1 | 256 | 300 | 7,406 | 62,818 | 18,565 | 8,792 | 7,874 | **5,417** | 11.6× | 3.4× |
| 2 | 8 | 300 | 198 | 1,123 | 404 | 294 | 243 | **222** | 5.1× | 1.8× |
| 2 | 64 | 300 | 1,960 | 9,969 | 2,608 | 2,082 | 1,769 | **1,257** | 7.9× | 2.1× |
| 2 | 256 | 300 | 7,406 | 44,865 | 11,333 | 8,870 | 7,593 | **4,996** | 9.0× | 2.3× |

Forward-only ratios (native v2 vs original) range from 8× to 17×; the full grid (d_h 128 and 300, forward and backward) is in `benchmarks/op_microbench.py` output.

## 3. Integration

Five methods are replaced in place (see {doc}`compatibility`); everything else, including `state_dict` keys and
initialisation, is untouched, and each replacement falls back per call to the original for GPU tensors and
dtypes other than float32/float64. The readout and aggregations use `index_add_`, which needs no expanded index either.

## 4. End-to-end ablation

| Configuration (bond MPNN, B=64, d_h=300, depth 3, 2 threads) | Training step, ms (median of rounds) | Range across rounds | Speedup |
| --- | ---: | ---: | ---: |
| Original Chemprop | 43.2 | 38.7–51.3 (n=14) | 1.00× |
| Original + fused Adam | 41.8 | 40.4–50.5 (n=3) | 1.03× |
| torch.compile | 45.2 | 45.0–45.3 (n=2) | 0.96× |
| accel: pure-PyTorch fallback | 19.6 | 18.7–22.6 (n=3) | 2.21× |
| accel: tiled-scatter variant (0) | 19.9 | 19.5–23.0 (n=3) | 2.17× |
| accel: CSR variant (1) | 20.2 | 20.2–23.8 (n=3) | 2.13× |
| chemprop_accel (default, variant 2) | 19.1 | 15.7–32.4 (n=11) | 2.26× |
| chemprop_accel + fused Adam | 20.6 | 18.2–21.7 (n=3) | 2.10× |
| chemprop_accel + compile | 31.0 | 30.5–31.5 (n=2) | 1.39× |

Reading this table honestly:

- Most of the end-to-end gain (≈2.2×) comes from **removing Chemprop's expanded-index scatter/gather pattern**, which the
  pure-PyTorch fallback already does. On this 2-vCPU machine the native kernel adds only a few percent at 2 threads
  (2.26× vs 2.21×) and about 14% at 1 thread (2.21× vs 1.93×), even though at op level it is 2–3× faster than the
  fallback (above): once the graph ops are cheap, the dense matrix products dominate (see the profile below).
  The native variants differ from each other by less than the round-to-round noise end to end. A machine with more
  cores or a slower memory system may show a different split; measure with `python -m chemprop_accel.bench --modes eager accel accel_pure`.
- All configurations produce the same predictions to ~1e-7 (see the equivalence table in the benchmark results).

## 5. Negative and neutral results

These were tried and **not** kept (or not kept as the default):

- **Native `scatter_sum` for the readout / mean / sum aggregations.** A native column-tiled and a CSR kernel were
  implemented; neither was reliably faster than PyTorch's `index_add_` (at 2 threads `index_add_` was often faster),
  so the native version was deleted and `index_add_` is used.
- **Column-tiled scatter (variant 0)** is the slowest native variant: it must zero and then reload an atom buffer and
  parallelises over only `d/32` tiles. It is retained for benchmarking and as a no-adjacency-construction reference.
- **Amortising/caching the CSR across message-passing layers.** Building the CSR costs ≈ 30 µs at E≈2,000 directed
  edges (about 8% of the op at d_h=300, more at small d_h). Re-using it across the `depth-1` layers and the backward pass would
  need a per-graph cache keyed on tensor identity, exactly the kind of stale-reference hazard the project avoids, for a
  ≲5% gain; per-call construction is kept.
- **Fused Adam (`fused=True`)** on CPU: no measurable benefit. Original: 41.8 ms with `fused=True` vs 43.2 ms without; accelerated: 20.6 ms vs 19.1 ms. The differences are inside the round-to-round range, and the optimiser is only ~4% of the step.
- **`torch.compile`:** `torch.compile` of the original model gave 0.96× (no gain) and of the accelerated model 1.39× vs original, i.e. *slower than the accelerated model in eager mode* (31.0 ms vs 19.1 ms). With `TORCH_LOGS=recompiles` we see Dynamo guard on the varying graph sizes and recompile for each new batch shape, and Chemprop's data-dependent `dim_size` in the aggregations (`batch.max()`) forces graph breaks, so on variable-sized molecular batches compilation costs a cold start (~1.4 s here) plus per-shape recompiles and mostly fails to pay back. Compiled execution is correct and supported (tested; predictions match), just not beneficial in this setting. Compile *after* importing the package.
- **Not implemented, not measured:** a persistent custom worker pool (the operations are short and already use
  PyTorch's pool; a second pool would oversubscribe), BF16/AMX kernels (requires hardware this project's CI does not
  have; bf16 tensors take the original code path), and larger fusions (linear + activation, aggregation + update):
  after this change the matrix multiplications are the dominant remaining cost and fusing around BLAS risks making
  them slower.

## 6. Remaining bottlenecks

After the change:

*accel=on · bond · B=64 · d_h=300 · threads=2 · total self-CPU 20.8 ms/step (profiler overhead included)*

| Operator | Self CPU (ms/step) | Share |
| --- | ---: | ---: |
| `aten::mm` | 11.06 | 53% |
| `aten::addmm` | 0.99 | 5% |
| `GeneratedBackwardFor_chemprop_accel_edge_message_default` | 0.98 | 5% |
| `aten::clamp_min` | 0.83 | 4% |
| `chemprop_accel::edge_message_backward` | 0.79 | 4% |
| `Optimizer.step#Adam.step` | 0.77 | 4% |
| `aten::threshold_backward` | 0.76 | 4% |
| `aten::add` | 0.66 | 3% |
| `aten::sqrt` | 0.35 | 2% |

(`GeneratedBackwardFor_…`/`edge_message_backward` are the fused backward; the entire graph-op cost is now about 9% of the step.)

What is left is dominated by dense linear algebra (`mm`/`addmm` and their backward), elementwise activation
(`threshold_backward`, `clamp_min`), `cat` and optimiser steps. Further wins would come from larger batches or
mixed precision on hardware with efficient low-precision GEMMs, not from further graph-op work.
