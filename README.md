# TotalUnimodularity.jl

A pure Julia implementation of the polynomial-time algorithm for testing
[total unimodularity](https://en.wikipedia.org/wiki/Unimodular_matrix#Total_unimodularity)
of integer matrices, based on Seymour's decomposition theorem.

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
cmr_is_totally_unimodular(M)                          # Seymour decomposition
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

See [THEORY.md](THEORY.md) for full mathematical details and
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for implementation
decisions and known issues.

## Performance

`is_totally_unimodular` works well for matrices up to approximately 8×10.
The Seymour decomposition step (Theorem 20.2) has O((m+n)^8) worst-case
complexity and becomes slow for larger matrices.

For matrices beyond the 12×12 decomposition-search threshold whose smaller
dimension is at most 24, the implementation switches to an exact
branch-and-prune Ghouila-Houri partition test — the matroid-intersection
search at those sizes would take hours to days, while the partition test
answers in milliseconds for typical inputs (worst case ~seconds up to
min-dimension 20, ~13s at 22). Only matrices whose smaller dimension
exceeds 24 fall back to the matroid-intersection search.

Inside the 12×12 decomposition search, candidate bipartitions are prefiltered
by a word-parallel GF(2) rank bound (rank mod 2 never exceeds rational rank),
which rejects the overwhelming majority of candidates in a few bit
operations — about 20× faster on hard 12×12 inputs.

### Benchmark: naive vs decomposition checker

Representative matrices, timed after JIT warmup (Linux x86-64, Julia 1.12).
The "path" column shows which stage of the algorithm decides the answer.

| Matrix | Size | TU | naive | `is_totally_unimodular` | deciding path |
|---|---|---|---|---|---|
| identity | 3×3 | true | 2 µs | 15 µs | reduced to empty |
| network matrix | 3×3 | true | 3 µs | 9 µs | network test |
| K₃₃ | 5×4 | true | 14 µs | 61 µs | network test |
| F₁ | 5×5 | true | 29 µs | 58 µs | special matrix (F₁/F₂) |
| R10 | 5×5 | true | 25 µs | 534 µs | special matrix (F₁/F₂) |
| R12 | 6×6 | true | 171 µs | 1.1 ms | decompose → pivot (Case 5/6) |
| Fano | 3×4 | false | 6 µs | 36 µs | Eulerian k≤3 filter |
| one_sum(K₃₃, K₃₃) | 10×8 | true | 11.7 ms | 86 µs | 1-sum component split |
| two_sum(K₃₃, K₃₃ᵈ) | 8×8 | true | 3.2 ms | 295 µs | decompose → 2-sum (Case 2/3) |
| CMR Eulerian test | 12×12 | false | 1.4 s | 2.3 s | decompose → 3-sum (Case 4) |
| CMR partition test | 14×14 | false | 15.7 s | 17.5 ms | Ghouila-Houri (>12×12) |
| 2-sum chain | 15×15 | true | — (infeasible) | 31 ms | Ghouila-Houri (>12×12) |
| 2-sum chain | 22×22 | true | — (infeasible) | 5.8 s | Ghouila-Houri (>12×12) |

For tiny matrices the naive checker wins on constant factors; the
decomposition checker pulls ahead from ~8×8 and remains usable far beyond
the naive checker's exponential wall. The 12×12 row is the worst case for
the current implementation: exactly at the decomposition-search size limit,
too small for the Ghouila-Houri routing.

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
`naive_is_totally_unimodular` on 2000 random matrices of size up to 5×6,
with no disagreements.

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