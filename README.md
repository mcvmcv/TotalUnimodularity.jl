# TotalUnimodularity.jl

A pure Julia implementation of total unimodularity testing for integer
matrices, following the recognition algorithm based on Seymour's
decomposition theorem.
See [total unimodularity](https://en.wikipedia.org/wiki/Unimodular_matrix#Total_unimodularity).

> **Note:** the algorithm of Schrijver's Theorem 20.3 is polynomial-time, but
> this implementation is not. 1-sums and 2-sums are split off in polynomial
> time, but a block that cannot be split that way is decided by an exact
> exponential-time test. See [Performance](#performance) for the practical
> limits.

## What is Total Unimodularity?

A matrix M with integer entries is **totally unimodular (TU)** if every square
submatrix has determinant in {-1, 0, 1}. Total unimodularity is important in
integer programming: if the constraint matrix of a linear program is TU, then
the LP relaxation always has an integer optimal solution.

Well-known examples of TU matrices include incidence matrices of bipartite
graphs and network matrices.

## Installation

> **Note:** This package is not yet registered in the Julia General Registry.
> To install directly from GitHub:
```julia
] add https://github.com/mcvmcv/TotalUnimodularity.jl
```

Once registered:
```julia
] add TotalUnimodularity
```

## Usage

### Testing for total unimodularity
```julia
using TotalUnimodularity

M = [1  0  1  0
     0  1  1  0
     0  0  1  1]

is_totally_unimodular(M)   # true
```

### Naive test (exponential time, for verification)
```julia
naive_is_totally_unimodular(M)   # true — checks all square submatrices
```

### CMR-style algorithms

`cmr_is_totally_unimodular` mirrors the CMR C library's `CMRtuTest` dispatcher
and exposes two additional exponential-time (but exact) criteria alongside the
default decomposition algorithm:

```julia
cmr_is_totally_unimodular(M)                          # Seymour decomposition (blocks ≤ 12×12)
cmr_is_totally_unimodular(M; algorithm=:eulerian)     # Camion's Eulerian criterion
cmr_is_totally_unimodular(M; algorithm=:partition)    # Ghouila-Houri criterion
```

### Seymour decomposition operations

If A and B are both TU, so are their 1-sum, 2-sum, and 3-sum:
```julia
A = [1 0 1; -1 1 0; 0 -1 -1]
B = [1 0; 0 1]

one_sum(A, B)    # block diagonal [A 0; 0 B]
two_sum(A, B)    # 2-sum (requires compatible border structure)
three_sum(A, B)  # 3-sum (requires compatible border structure)
```

### Special matrices
```julia
F_1    # first Seymour special matrix (5×5 TU, non-network)
F_2    # second Seymour special matrix (5×5 TU, non-network)
```

## Algorithm

The implementation follows Schrijver's *Theory of Linear and Integer
Programming* (Chapters 19–20), implementing Theorem 20.3:

1. **Preprocessing:** Remove rows/columns with ≤1 nonzero and linearly
   dependent rows/columns (equal or opposite pairs).
2. **Network matrix test:** Test whether M or its transpose is a network
   matrix (Theorem 20.1).
3. **Special matrix test:** Test whether M is equivalent to F_1 or F_2
   under row/column permutations and ±1 scalings.
4. **Seymour decomposition:** Find a partition M = [A B; C D] with
   rank(B) + rank(C) ≤ 2 (Theorem 20.2), then recurse on the six cases
   of Theorem 20.3.

By default, step 4 is carried out differently: 2-separations
(rank(B) + rank(C) ≤ 1) are found by a polynomial search and split off, and a
block with no 2-separation is decided by an exact Ghouila-Houri partition
test, because that is faster in practice than the general separation search;
see [Performance](#performance).

See [THEORY.md](THEORY.md) for full mathematical details and
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for implementation
decisions and known issues.

## Performance

The cost of `is_totally_unimodular` is governed by the largest block that
is left after all polynomial-time steps, not by the size of the input. Those
steps are: reduction, splitting into connected components (1-sums), the
network and special-matrix tests, and splitting along 2-separations (2-sums),
each applied again to the pieces. A matrix built from small pieces by 1- and
2-sums is therefore decided quickly at any size, as is any network matrix or
transpose of one.

A block that survives all of that — it has no 2-separation and is neither a
network matrix, the transpose of one, nor F_1/F_2 — first passes a cheap
Eulerian pre-filter and is then routed by size:

- **Smaller dimension ≤ 24:** an exact branch-and-prune Ghouila-Houri
  partition test. It is exponential in the smaller dimension, but answers in
  milliseconds for typical inputs (worst case ~seconds up to min-dimension
  20, ~13s at 22).
- **Both dimensions > 24:** no practical route. Up to 64 rows plus columns
  the block goes to the Seymour decomposition with the matroid-intersection
  separation search of Theorem 20.2, O((m+n)^8); beyond that, to the
  Ghouila-Houri test. Both are impractically slow at these sizes.

So the practical limit is a smaller dimension of about 22 *for such a block*.
The remaining gap to a polynomial algorithm is the search for 3-separations
(3-sums), which is still exhaustive.

`cmr_is_totally_unimodular(M; algorithm=:decomposition)` instead runs the
Seymour decomposition on blocks up to 12×12 (without the 2-sum split), finding
the separation by
enumerating all row/column bipartitions with a word-parallel GF(2) rank
prefilter (rank mod 2 never exceeds rational rank). It gives the same
answers but is slower — about 4× in aggregate on composed inputs, and 2.2 s
against 3 ms on the hardest 12×12 input below.

### Benchmark: naive vs `is_totally_unimodular`

Representative matrices, timed after JIT warmup (Linux x86-64, Julia 1.12).
The "path" column shows which stage of the algorithm decides the answer.

| Matrix | Size | TU | naive | `is_totally_unimodular` | deciding path |
|---|---|---|---|---|---|
| identity | 3×3 | true | 2 µs | 15 µs | reduced to empty |
| network matrix | 3×3 | true | 3 µs | 9 µs | network test |
| K₃₃ | 5×4 | true | 14 µs | 61 µs | network test |
| F₁ | 5×5 | true | 29 µs | 58 µs | special matrix (F₁/F₂) |
| R10 | 5×5 | true | 25 µs | 534 µs | special matrix (F₁/F₂) |
| R12 | 6×6 | true | 171 µs | 83 µs | Ghouila-Houri |
| Fano | 3×4 | false | 6 µs | 36 µs | Eulerian k≤3 filter |
| one_sum(K₃₃, K₃₃) | 10×8 | true | 11.7 ms | 86 µs | 1-sum component split |
| two_sum(K₃₃, K₃₃ᵈ) | 8×8 | true | 3.2 ms | 116 µs | 2-sum split |
| CMR Eulerian test | 12×12 | false | 1.4 s | 2.5 ms | Ghouila-Houri |
| CMR partition test | 14×14 | false | 15.7 s | 13.4 ms | Ghouila-Houri |
| 2-sum chain | 15×15 | true | — (infeasible) | 0.45 ms | 2-sum split |
| 2-sum chain | 22×22 | true | — (infeasible) | 1.0 ms | 2-sum split |
| 2-sum chain | 145×144 | true | — (infeasible) | 138 ms | 2-sum split |
| 2-sum chain | 705×704 | true | — (infeasible) | 59 s | 2-sum split |

For tiny matrices the naive checker wins on constant factors;
`is_totally_unimodular` pulls ahead from ~6×6 and remains usable far beyond
the naive checker's exponential wall.

### Benchmark: CMR vs `is_totally_unimodular`

[CMR](https://github.com/discopt/cmr) is a C library with a polynomial-time
implementation of the full Seymour decomposition. The table compares its
`cmr-tu` tool (Release build, default decomposition algorithm) with
`is_totally_unimodular` on the same machine. CMR times are its own reported
recognition time, excluding process start and file reading; Julia times are
after JIT warmup. Both gave the same answer on every matrix where both
finished.

| Matrix | Size | TU | CMR | `is_totally_unimodular` |
|---|---|---|---|---|
| K₃₃ | 5×4 | true | 0.39 ms | 0.018 ms |
| F₁ | 5×5 | true | 0.47 ms | 0.015 ms |
| R10 | 5×5 | true | 0.32 ms | 0.50 ms |
| R12 | 6×6 | true | 1.02 ms | 0.096 ms |
| Fano | 3×4 | false | 0.35 ms | 0.012 ms |
| CMR Eulerian test | 12×12 | false | 2.8 ms | 2.5 ms |
| CMR partition test | 14×14 | false | 3.9 ms | 13.4 ms |
| 2-sum chain, 6 blocks | 22×22 | true | 1.5 ms | 0.98 ms |
| 2-sum chain, 13 blocks | 47×46 | true | 3.2 ms | 5.1 ms |
| 2-sum chain, 41 blocks | 145×144 | true | 12.5 ms | 138 ms |
| 2-sum chain, 201 blocks | 705×704 | true | 0.21 s | 58.8 s |
| 2-sum chain (201 blocks) + non-TU block | 708×706 | false | 0.22 s | 58.9 s |
| random network | 20×40 | true | 0.58 ms | 0.35 ms |
| random network | 50×100 | true | 0.95 ms | 3.2 ms |
| random network | 100×200 | true | 1.5 ms | 25.8 ms |
| random network | 200×400 | true | 3.0 ms | 133 ms |
| network, one entry flipped | 50×100 | false | 147 ms | 24.3 ms |
| random sparse, density 0.1 | 30×30 | false | 1.9 ms | 1.1 ms |
| random sparse, density 0.05 | 100×100 | false | 10.0 ms | 101 ms |
| random sparse, density 0.5 | 20×20 | false | 0.43 ms | 1.9 ms |

The next family has no 2-separation and is neither a network matrix nor the
transpose of one: the 3-sum of the network matrix of Kₙ with the transpose
of the network matrix of Kₘ plus one vertex of degree 3 (written Kₘ* below),
with rows and columns permuted and rescaled. This is where
`is_totally_unimodular` falls back to its exponential test.

| Matrix | Size | TU | CMR | `is_totally_unimodular` |
|---|---|---|---|---|
| K₅ ⊕₃ K₅* | 10×8 | true | 1.3 ms | 0.23 ms |
| K₆ ⊕₃ K₅* | 11×12 | true | 1.5 ms | 1.7 ms |
| K₆ ⊕₃ K₆* | 15×13 | true | 1.9 ms | 7.4 ms |
| K₇ ⊕₃ K₆* | 16×18 | true | 1.3 ms | 66 ms |
| K₇ ⊕₃ K₇* | 21×19 | true | 1.1 ms | 0.64 s |
| K₈ ⊕₃ K₇* | 22×25 | true | 2.9 ms | 8.3 s |
| K₈ ⊕₃ K₈* | 28×26 | true | 1.4 ms | > 3 min (stopped) |
| K₉ ⊕₃ K₉* | 36×34 | true | 7.1 ms | > 3 min (stopped) |
| K₁₂ ⊕₃ K₁₂* | 66×64 | true | 40.5 ms | > 3 min (stopped) |
| the nine above, one entry flipped | 10×8 – 66×64 | false | 0.4 – 39 ms | 0.09 – 20 ms |

In short: on small matrices this package wins on constant factors; on large
matrices that split into small pieces or are network matrices it finishes
but CMR scales much better (the 2-separation search restarts from scratch
after every split); and on large TU blocks with no 2-separation it is
unusable beyond a smaller dimension of about 22, while CMR stays in
milliseconds. Non-TU inputs are usually cheap for both. If you need large
general instances, use CMR.

Rank computations avoid floating-point SVD entirely: the hot paths use
Float64 Gaussian elimination (exact for the small {-1,0,1} matrices arising
here) with rank caching, and exact Bareiss integer elimination elsewhere.
`is_totally_unimodular` accepts any `AbstractMatrix` with integer-valued
entries.

For verification on small matrices, `naive_is_totally_unimodular` is
available but has exponential time complexity.

## Testing
```julia
] test TotalUnimodularity
```

The test suite verifies `is_totally_unimodular` against
`naive_is_totally_unimodular` on 2000 random matrices of size up to 5×6, and
against the exact Ghouila-Houri test on larger random matrices and on 2000
structured inputs (sums of F_1, F_2, K₃,₃ and network matrices with random
pivots and scalings) that exercise the decomposition cases. The 2-separation
search is checked against brute-force enumeration on small matrices, and
2-sum chains up to 145×144 are tested end to end.

## Background

Total unimodularity testing is based on Seymour's decomposition theorem
(Seymour 1980): a matrix is TU if and only if it can be constructed from
network matrices, their transposes, F_1, and F_2 via 1-sums, 2-sums, and
3-sums. This characterisation leads to a polynomial-time recognition
algorithm described in Schrijver (1986).

This package provides the only known pure Julia implementation of this
algorithm.

## References

- Schrijver, A. (1986). *Theory of Linear and Integer Programming*. Wiley.
  Chapters 19–20.
- Seymour, P.D. (1980). Decomposition of regular matroids. *Journal of
  Combinatorial Theory, Series B*, 28(3), 305–359.
- Ghouila-Houri, A. (1962). Caractérisation des matrices totalement
  unimodulaires. *Comptes Rendus de l'Académie des Sciences*, 254,
  1192–1194.

## Authors

- Michael McVeagh (mcvmcv)
- Claude (Anthropic) — AI pair programming assistant

## License

MIT