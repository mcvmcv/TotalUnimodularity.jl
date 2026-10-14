# TotalUnimodularity.jl

A pure Julia implementation of total unimodularity testing for integer
matrices, following the recognition algorithm based on Seymour's
decomposition theorem.
See [total unimodularity](https://en.wikipedia.org/wiki/Unimodular_matrix#Total_unimodularity).

> **Note:** every step on the default route is polynomial, but with a higher
> degree than the best known algorithms, and an exact exponential-time test
> is kept for very small blocks and as a fallback. On the hardest large
> inputs the [CMR](https://github.com/discopt/cmr) C library is still
> faster, and it can return the decomposition and certificates, where this
> package only answers yes or no; see [Performance](#performance) for
> measured limits and the comparison.

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

The partition in step 4 is not found with the general search of Theorem
20.2, which is far too slow in practice. 2-separations
(rank(B) + rank(C) ≤ 1) and 3-separations (rank(B) + rank(C) = 2) are each
found by fixing a few entries of B and C, after which the side of every
other row and column follows from single-premise rules, and a split is a
closed set in a digraph; see [THEORY.md](THEORY.md). Blocks whose smaller
dimension is at most 8 are decided by an exact Ghouila-Houri partition test
instead, which is faster at that size.

See [THEORY.md](THEORY.md) for full mathematical details and
[IMPLEMENTATION_NOTES.md](IMPLEMENTATION_NOTES.md) for implementation
decisions and known issues.

## Performance

`is_totally_unimodular` reduces the matrix, splits it into connected
components (1-sums), tests each block for being a network matrix, the
transpose of one, or F_1/F_2, and otherwise splits it along a 2-separation
(2-sum) or, failing that, a 3-separation (3-sum), applying the same steps to
the pieces. A block with no 2-separation and no 3-separation that is none of
the basic types is not TU, by Seymour's theorem. All of these steps are
polynomial; the most expensive is the 3-separation search, O((m+n)² · m · n)
when no separation exists.

The exact Ghouila-Houri partition test, exponential in the smaller
dimension, remains in two places: for blocks whose smaller dimension is at
most 8, where it is the faster option, and as a fallback when the
decomposition cannot proceed (a repeated matrix on the recursion path, more
than 100 nested pivots, or a 3-sum whose sign cannot be determined). The
package counts these fallbacks, and the test suite checks that none occurs.

In practice: matrices that split into small pieces and network matrices
are decided in milliseconds to a tenth of a second at sizes of several
hundred rows. Blocks that need 3-separations cost more: a few milliseconds
at 66×64, a few tenths of a second at 276×274. The slowest inputs are large
matrices that are not TU but pass the cheap pre-filter, where the
3-separation search has to try every candidate: about 0.16 s at 66×64, 10 s
at 276×274.

`cmr_is_totally_unimodular(M; algorithm=:decomposition)` instead runs the
Seymour decomposition on blocks up to 12×12 with an exhaustive search for
the separation: all row/column bipartitions, with a word-parallel GF(2)
rank prefilter (rank mod 2 never exceeds rational rank). It gives the same
answers and exists as an independent check on the default route; it is
slower, 2.2 s against a few milliseconds on the hardest 12×12 input below.

### Benchmark: naive vs `is_totally_unimodular`

Representative matrices, timed after JIT warmup (Linux x86-64, Julia 1.12).
The "path" column shows which stage of the algorithm decides the answer.

| Matrix | Size | TU | naive | `is_totally_unimodular` | deciding path |
|---|---|---|---|---|---|
| identity | 3×3 | true | 2 µs | 3.8 µs | reduced to empty |
| network matrix | 3×3 | true | 3 µs | 2.6 µs | network test |
| K₃₃ | 5×4 | true | 14 µs | 12 µs | network test |
| F₁ | 5×5 | true | 29 µs | 8.7 µs | special matrix (F₁/F₂) |
| R10 | 5×5 | true | 25 µs | 487 µs | special matrix (F₁/F₂) |
| R12 | 6×6 | true | 171 µs | 77 µs | Ghouila-Houri |
| Fano | 3×4 | false | 6 µs | 11 µs | 2×2/3×3 pre-filter |
| one_sum(K₃₃, K₃₃) | 10×8 | true | 11.7 ms | 26 µs | 1-sum component split |
| two_sum(K₃₃, K₃₃ᵈ) | 8×8 | true | 3.2 ms | 61 µs | 2-sum split |
| CMR Eulerian test | 12×12 | false | 1.4 s | 1.0 ms | no 3-separation |
| CMR partition test | 14×14 | false | 15.7 s | 1.7 ms | no 3-separation |
| 2-sum chain | 15×15 | true | — (infeasible) | 0.24 ms | 2-sum split |
| 2-sum chain | 22×22 | true | — (infeasible) | 0.39 ms | 2-sum split |
| 2-sum chain | 145×144 | true | — (infeasible) | 6.5 ms | 2-sum split |
| 2-sum chain | 705×704 | true | — (infeasible) | 0.09 – 0.13 s | 2-sum split |

The naive checker and `is_totally_unimodular` are level on the very
smallest matrices; `is_totally_unimodular` pulls ahead from about 5×5 (R10
is the exception) and remains usable far beyond the naive checker's
exponential wall. The naive column was measured earlier than the other.

### Benchmark: CMR vs `is_totally_unimodular`

[CMR](https://github.com/discopt/cmr) is a C library with a polynomial-time
implementation of the full Seymour decomposition, a simplified version of
Truemper's algorithm. It can also return the decomposition tree and
certificates, where this package only answers yes or no. On the inputs
below the two are within a small factor of each other in either direction,
and on the hardest large inputs, non-TU matrices that pass the pre-filter,
CMR is faster (see [Larger matrices](#larger-matrices)). These are a few
families of test matrices, not a survey: other inputs may favour either
program.

The tables compare CMR's `cmr-tu` tool (Release build, default decomposition
algorithm) with `is_totally_unimodular` on the same machine. Both gave the
same answer on every matrix. `benchmark/run.sh` reproduces them.

The two columns are not measured the same way, and the difference favours
this package on the smallest rows:

- CMR: one run in a fresh process, using the recognition time it reports
  itself (process start and file reading excluded).
- This package: the best of several runs in one warm process (21 runs for
  times under 50 ms, 4 up to 2 s, 1 above), after compilation.

CMR also has a fixed cost of about 0.3 ms per call that does not shrink
with the matrix (0.30–0.34 ms over ten runs on the 5×4 matrix), which is
what the first rows of the table mostly show.

| Matrix | Size | TU | CMR | `is_totally_unimodular` |
|---|---|---|---|---|
| K₃₃ | 5×4 | true | 0.34 ms | 0.013 ms |
| F₁ | 5×5 | true | 0.33 ms | 0.009 ms |
| R10 | 5×5 | true | 0.34 ms | 0.48 ms |
| R12 | 6×6 | true | 0.95 ms | 0.066 ms |
| Fano | 3×4 | false | 0.35 ms | 0.013 ms |
| CMR Eulerian test | 12×12 | false | 3.0 ms | 0.96 ms |
| CMR partition test | 14×14 | false | 3.7 ms | 1.7 ms |
| 2-sum chain, 6 blocks | 22×22 | true | 1.5 ms | 0.38 ms |
| 2-sum chain, 13 blocks | 47×46 | true | 3.2 ms | 1.2 ms |
| 2-sum chain, 41 blocks | 145×144 | true | 13.4 ms | 6.2 ms |
| 2-sum chain, 201 blocks | 705×704 | true | 0.22 s | 0.09 – 0.13 s |
| 2-sum chain (13 blocks) + non-TU block | 50×48 | false | 3.3 ms | 1.3 ms |
| 2-sum chain (201 blocks) + non-TU block | 708×706 | false | 0.22 s | 0.12 – 0.13 s |
| random network | 20×40 | true | 0.62 ms | 0.15 ms |
| random network | 50×100 | true | 1.0 ms | 0.77 ms |
| random network | 100×200 | true | 1.5 ms | 2.6 ms |
| random network | 200×400 | true | 2.9 ms | 7.6 ms |
| network, one entry flipped | 50×100 | false | 0.15 s | 2.5 ms |
| random sparse, density 0.1 | 30×30 | false | 1.9 ms | 0.29 ms |
| random sparse, density 0.05 | 100×100 | false | 10.0 ms | 4.0 ms |
| random sparse, density 0.5 | 20×20 | false | 0.44 ms | 0.35 ms |

The next family has no 2-separation and is neither a network matrix nor the
transpose of one: the 3-sum of the network matrix of Kₙ with the transpose
of the network matrix of Kₘ plus one vertex of degree 3 (written Kₘ* below),
with rows and columns permuted and rescaled. These are decided through the
3-separation search; before it existed, the 22×25 member took 7.5 s and the
larger ones did not finish.

| Matrix | Size | TU | CMR | `is_totally_unimodular` |
|---|---|---|---|---|
| K₅ ⊕₃ K₅* | 10×8 | true | 0.76 ms | 0.17 ms |
| K₆ ⊕₃ K₅* | 11×12 | true | 1.5 ms | 0.29 ms |
| K₆ ⊕₃ K₆* | 15×13 | true | 1.9 ms | 0.32 ms |
| K₇ ⊕₃ K₆* | 16×18 | true | 1.2 ms | 0.43 ms |
| K₇ ⊕₃ K₇* | 21×19 | true | 1.1 ms | 0.36 ms |
| K₈ ⊕₃ K₇* | 22×25 | true | 3.0 ms | 0.71 ms |
| K₈ ⊕₃ K₈* | 28×26 | true | 1.6 ms | 0.72 ms |
| K₉ ⊕₃ K₉* | 36×34 | true | 6.7 ms | 1.9 ms |
| K₁₂ ⊕₃ K₁₂* | 66×64 | true | 39.2 ms | 3.2 ms |
| the nine above, one entry flipped | 10×8 – 66×64 | false | 0.4 – 38 ms | 0.04 – 1.4 ms |

How to read these numbers:

- **Small matrices** (up to about 20×20): this package is usually faster,
  by one to two orders of magnitude on the smallest, because of CMR's fixed
  cost per call. If you test many small matrices from Julia, that matters.
- **Non-TU matrices with a small violation**: this package is usually
  faster, because a cheap pre-filter finds any 2×2 or 3×3 violating
  submatrix before the decomposition starts. On "network, one entry
  flipped" and the flipped 66×64 matrix, CMR spends its time enumerating
  3-separation candidates instead.
- **Larger TU matrices**: within a small factor of each other, in either
  direction: CMR is about 2.5× ahead on the 200×400 network matrix, this
  package 3.5× to 12× on the 36×34 and 66×64 3-sums.

#### Larger matrices

The same 3-sum family at larger sizes (`benchmark/scaling.jl`; single runs,
so read the ratios as rough): each member as built, after six random
pivots, and with one entry changed so that it is not TU but still passes
the pre-filter, which is the worst case for both programs.

| Size | TU: CMR | TU: this package | non-TU: CMR | non-TU: this package |
|---|---|---|---|---|
| 36×34 | 1.8 – 1.9 ms | 1.0 – 3.5 ms | 13 – 14 ms | 27 ms |
| 66×64 | 13 – 16 ms | 4.8 – 6.2 ms | 59 – 72 ms | 0.16 s |
| 120×118 | 31 – 195 ms | 98 – 133 ms | 0.50 s | 0.83 – 0.85 s |
| 190×188 | 0.12 – 1.6 s | 49 – 85 ms | 2.5 s | 3.5 – 4.0 s |
| 276×274 | 0.26 – 0.38 s | 0.27 – 0.50 s | 9.4 – 9.5 s | 10 – 11 s |
| 378×376 | 0.41 – 0.76 s | 0.40 – 0.57 s | 18 s | 23 – 24 s |

On TU matrices the two are level within a factor of about 3 either way,
with no trend in size; the search here stops at the first separation it
finds, so its time depends on luck in the order it tries candidates. On the
non-TU worst case CMR is ahead, 1.6× to 2.7× up to 120×118 and 1.1× to
1.6× above, and both become slow: about 20 s at 378×376.

So: this package is a reasonable choice at these sizes and needs no C
toolchain. Use CMR when the hardest non-TU inputs dominate your workload,
when you need the decomposition or a certificate, or at sizes beyond those
measured here.

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
pivots and scalings) that exercise the decomposition cases. The 2- and
3-separation searches are checked against brute-force enumeration on small
matrices; 2-sum chains up to 705×704 and 3-sums up to 66×64 are tested end
to end, and 1500 pivoted, trimmed and perturbed 3-sums are compared with the
exact test.

Matrices too large for the exact test can be checked against CMR with
`benchmark/fuzz_cmr.jl`, which needs the `cmr-tu` binary and is therefore
not part of the suite.

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
- Walter, M. and Truemper, K. (2013). Implementation of a unimodularity
  test. *Mathematical Programming Computation*, 5(1), 57–73. The method
  behind CMR's total unimodularity test.
- CMR — Combinatorial Matrix Recognition, a C library by Matthias Walter:
  <https://github.com/discopt/cmr> (MIT license). The
  `cmr_is_totally_unimodular` interface mirrors its `CMRtuTest`, part of
  this package's test suite is ported from its `test_tu.cpp`, and the
  benchmarks above compare against it.

## Authors

- Michael McVeagh (mcvmcv)
- Claude (Anthropic) — AI pair programming assistant

## License

MIT