# Implementation Notes

This document records key decisions, bugs found, and fixes applied during
development of the TotalUnimodularity.jl package. Intended as context for
future developers and AI assistants working on this codebase.

## Development History

The package was implemented incrementally following Schrijver Chapter 19-20,
with each component tested against `naive_is_totally_unimodular` as an oracle.

## Key Design Decisions

### `_reduce` returns `Tuple{Bool, Matrix{Int}}`
Rather than returning `Union{Matrix{Int}, Nothing}`, `_reduce` returns a
`(Bool, Matrix{Int})` tuple for type stability. The `Bool` indicates whether
all entries are in {-1, 0, 1}. This pattern is used consistently throughout.

### `_extract_rank1` does NOT normalise, and g is signed
Early versions normalised f to have positive first entry (to ensure g is
{0,+1}). This was wrong — negating f also negates the relationship between
f and the columns of B, breaking the factorisation. The correct implementation
takes f as the first nonzero column of B without any normalisation.

Both f and g are genuinely {0,±1}: a rank-1 {-1,0,1} matrix B = f⊗g can have
columns equal to -f (where g[j] = -1). An earlier version assumed g ∈ {0,+1}
and silently produced f⊗g ≠ B for such blocks, causing false negatives (e.g.
a 2-sum whose glue row has mixed signs). `_extract_rank1` now sets
g[j] ∈ {+1,-1,0} according to whether column j equals f, -f, or 0, and Case 4
normalises the columns of A and D by g (in addition to normalising rows by f)
to reach Schrijver's standard form (28).

### Case 4 normalisation
Before applying the Case 4 formula, A and D must be normalised so that B
and C are in standard form [0; 1_block] and [1_block 0] respectively.
This is done by multiplying rows of A in B_rows by f_B[i], and rows of D
in C_rows by f_C[i]. Failing to do this normalisation caused false positives
(non-TU matrices being reported as TU).

### Case 4 non-degeneracy check
`_decompose` may return a decomposition where the A sub-block has linearly
dependent rows or columns (even though M itself does not). Schrijver's Case 4
proof assumes A and D are "generic" — applying the formula to degenerate A or D
gives wrong results. The fix: check A and D for trivial/dependent vectors before
applying Case 4, and return `false` if degenerate.

This is a defensive fix — it may give false negatives for some TU matrices,
but in practice all tested matrices agree with `naive_is_totally_unimodular`.

### Cycle detection in Cases 5 and 6
Cases 5 and 6 pivot on a nonzero entry of B (or C) and recurse. The pivot
does not always produce a matrix that decomposes as Case 4 — instead,
`_decompose` may find another Case 5 decomposition, leading to an infinite
loop through a cycle of 4 equivalent matrices.

The fix: maintain a `Set{Matrix{Int}}` of the matrices on the *current
recursion path*. If the same matrix is encountered again on the same path,
the decomposition has made no progress, so the matrix is decided exactly
with `_tu_partition` instead. Each matrix is removed from the set when its
subtree completes — an earlier version kept every visited matrix forever,
which wrongly rejected identical matrices appearing in sibling branches
(e.g. `one_sum(K33, K33)` returned `false` even though it is TU).

A cycle says nothing about total unimodularity. Until October 2026 the
guard returned `false`, which produced false negatives on TU matrices (see
Bug 8).

### `_is_trivial_vector` definition
The initial implementation checked `count(!iszero, v) == 1 && all(x -> x in (0,1), v)`,
which missed vectors like [0, 0, -1] — these have one nonzero but it's -1, not +1.
Schrijver says to delete rows/columns with "at most one nonzero" regardless of sign.

The correct definition is simply `count(!iszero, v) <= 1`.

This bug caused `_reduce` to fail to drop columns with a single -1 entry,
allowing non-TU matrices to pass through the preprocessing step uncaught.

### `_build_h` U index bug
The initial implementation of `_build_h` checked U_k against W_i for i ∈ C_k
when determining adjacency. The correct condition (from Schrijver equation 4) is:
- Check U_l (the OTHER component's U) against W_i for i ∈ C_k
- Check U_k against W_j for j ∈ C_l

This caused H to have no edges for F_2, making it trivially bipartite and
incorrectly identifying F_2 as a network matrix.

## Bugs Found During Random Testing

All bugs below were found by comparing `is_totally_unimodular` against
`naive_is_totally_unimodular` on random {-1, 0, 1} matrices.

### Bug 1: `_is_trivial_vector` missing negative standard basis vectors
**Symptom:** Non-TU matrices reported as TU.
**Cause:** Columns like [0, 0, -1] not recognised as trivial, so not dropped
by `_reduce`. The un-reduced matrix then passed through Case 4 incorrectly.
**Fix:** Simplified `_is_trivial_vector` to `count(!iszero, v) <= 1`.

### Bug 2: Case 4 applied without normalisation
**Symptom:** Non-TU matrices reported as TU.
**Cause:** B = f_B⊗g_B with f_B having negative entries. Without normalising
A (scaling B_rows by f_B[i]), the A sub-blocks A1,A2,A3,A4 did not correspond
to Schrijver's standard form (28). The matrices in (31) were constructed from
the wrong sub-blocks.
**Fix:** Normalise A and D before partitioning into sub-blocks.

### Bug 3: Case 4 applied to degenerate decomposition
**Symptom:** Non-TU matrices reported as TU.
**Cause:** `_decompose` found a partition where A had linearly dependent rows
(e.g. A = [-1,-1; 1,1]). The matrices in (31) constructed from this degenerate
A were both TU even though M was not.
**Fix:** Check A and D for degeneracy before applying Case 4; return `false`
if degenerate.

### Bug 4: Cases 5/6 infinite loop
**Symptom:** Stack overflow / maximum recursion depth exceeded on some
non-TU matrices.
**Cause:** After the Case 5 pivot, `_decompose` found another Case 5
decomposition (rather than Case 4 as Schrijver guarantees), leading to
a cycle of 4 equivalent matrices.
**Fix:** Cycle detection via `seen::Set{Matrix{Int}}`.

### Bug 5: `_extract_rank1` normalisation error
**Symptom:** `f * g ≠ B` for some rank-1 matrices.
**Cause:** After negating f to make first entry positive, columns of B
equal -f (the original f) rather than f (the negated f). The check
`B[:,j] == f` failed for these columns, giving g = [0,0,...,0].
**Fix:** Removed normalisation from `_extract_rank1`.

### Bug 6: global `seen` set rejected duplicate sibling blocks
**Symptom:** TU matrices with repeated blocks reported as non-TU, e.g.
`one_sum(K33, K33)` → `false`.
**Cause:** Cycle detection kept every visited matrix in `seen` forever, so
the second occurrence of an identical block in a *sibling* branch (not a
cycle) hit `M in seen → return false`.
**Fix:** Path-based cycle detection — matrices are removed from `seen` when
their subtree completes.

### Bug 7: `_extract_rank1` assumed g ∈ {0,+1}
**Symptom:** False negatives for 2-sums whose glue row has mixed signs, e.g.
`two_sum(K33, K33d')` with a negated column of K33d.
**Cause:** For B = f⊗g with g[j] = -1, column j of B equals -f; the old code
set g[j] = 0, producing f⊗g ≠ B, and the Case 2/3/4 recursion then tested the
wrong matrices.
**Fix:** g[j] ∈ {+1,-1,0} by matching columns against ±f; Case 4 additionally
normalises columns by g (see design note above).

### Bug 8: pivot-cycle guard reported TU matrices as non-TU
**Symptom:** About 1 in 800 composed inputs (2-sums of F_1/F_2/K33/network
blocks with random pivots and scalings) that are TU returned `false`.
**Cause:** `_apply_decomposition` alternated between the degenerate 3-sum
retry and a rank-2 pivot (Cases 5/6) and came back to a matrix already on
the recursion path; the guard treated that as non-TU.
**Fix:** on a cycle, decide the matrix with `_tu_partition`.

### Bug 9: second Case 4 matrix assembled with misaligned blocks
**Symptom:** About 1 in 6000 composed inputs that are TU returned `false`.
**Cause:** the first row of `mat2` was written with block widths
1, nBK, nnotBK, 1 while the rows below use 1, 1, nBK, nnotBK. The totals
match, so the concatenation succeeded, but the 1s sat over the wrong columns
of D whenever nBK ≠ nnotBK.
**Fix:** `[ε₁ 0 ones(1,nBK) zeros(1,nnotBK)]`.

Both were invisible to uniform random testing and to any single fuzz seed
of a few thousand inputs; see "Structured fuzz" under Testing.

## Known Limitations

### Performance and routing
1-sums and 2-sums are split off in polynomial time; what remains is not
polynomial in practice. `_is_tu_irreducible` runs, in order:

1. the network, transpose-network and special-matrix tests;
2. `_find_two_separation` (see below) — on success the block is split with
   `_apply_decomposition` (Case 2) and both pieces recurse from the top;
3. the Eulerian k ≤ 3 pre-filter. It is O(m³n³) in the worst case, which is
   why it runs after the split: on a splittable matrix only the pieces pay
   for it;
4. routing of the block, which now has no 2-separation, by size:
   - min(m,n) ≤ 24: the exact branch-and-prune Ghouila-Houri `_tu_partition`
     test. Measured worst cases (dense TU inputs, which force full
     exhaustion): ~0.1s at min-dim 16, ~0.5s at 18, ~3s at 20, ~13s at 22;
     non-TU inputs usually exit in milliseconds.
   - both dimensions > 24 and m + n ≤ 64: `_decompose` →
     `_decompose_matroid`, O((m+n)^8). It is effectively unusable: for a
     14×14 matrix it enumerates ~3.4×10⁸ (S,T) pairs and runs for hours. Its
     Float64 rank routine is also only argued exact up to ~20 rows.
   - both dimensions > 24 and m + n > 64: `_tu_partition`. Not because it is
     feasible there, but because `_decompose_matroid` keeps element sets in
     `UInt64` masks and cannot represent the block at all.

The exhaustive ≤12×12 bipartition search in `_decompose` is used only with
`fast = false`, i.e. `cmr_is_totally_unimodular(M; algorithm=:decomposition)`,
which also skips step 2 for those blocks so that Cases 2–6 stay reachable
through the general search. The default route skips it because
`_tu_partition` is faster at every size measured: ~4× in aggregate on
composed inputs, and 3 ms against 2.2 s on the 12×12 CMR Eulerian test
matrix. The decomposition code is kept correct by running every oracle and
regression test through both routes.

Making the 3-separation search scale (Truemper/CMR-style) is the main open
performance problem: a large block with no 2-separation that is neither a
network matrix nor the transpose of one still has no practical route.

### 2-separation search (`_find_two_separation`)
A 2-separation of a connected matrix is a split with one cross block zero
and the other of rank 1. Fix a nonzero entry (i0, j0) of the rank-1 block
M[R1, C2]; then "belongs to side 1" is closed under two single-premise
rules — a column on side 1 pulls in every row where it is nonzero (keeps
M[R2, C1] = 0), and a row r on side 1 pulls in every column c whose 2×2
minor on rows {i0, r}, columns {j0, c} is nonzero (keeps M[R1, C2] rank 1).
The rules are the edges of a digraph on the other m+n-2 rows and columns,
and a 2-separation with that pivot exists iff the digraph is not strongly
connected: any nonempty proper closed set is a valid side 1. Three searches
per pivot decide that (forward from a node, backward to it, forward from a
node that cannot reach it), so the whole search is O(nnz · m · n).

The function is specified for reduced, connected matrices, which is what
`_is_tu_irreducible` passes it; the unit test checks it against brute-force
enumeration of all splits under the same precondition.

### Recursion depth counts pivots only
`depth` used to be incremented on every recursive call, with a cutoff at
100 that falls back to `_tu_partition`. A 2-sum chain of k blocks recurses
k deep, so a chain of more than 100 blocks would hit the cutoff and hand a
huge matrix to the exponential test. The sum cases (1–4) always recurse on strictly smaller
matrices and cannot run away, so only the pivot cases (5/6) increment
`depth` now.

### Eulerian filter skips zero columns
For a fixed row subset, `_tu_eulerian` only considers columns with an even
and *nonzero* number of nonzeros in those rows. A minimal non-TU submatrix
is nonsingular, so it has no zero column; the criterion stays exact and
sparse matrices lose most of their candidate columns.

### Ghouila-Houri: keep subset choice and sign search separate
`_tu_partition` enumerates subsets in an outer phase and searches signs in an
inner branch-and-prune phase. The two must NOT be interleaved into a single
in/out/± tree: subsets sharing an in/out prefix would be forced to share
sign choices along it, computing a strictly stronger condition than
Ghouila-Houri (each subset gets its own signing) and yielding false
negatives. An interleaved variant passed 20,000 uniform random oracle tests
before a density-biased fuzz exposed it — uniform random {-1,0,1} draws are
a weak test distribution for this algorithm.

Two sound accelerations inside the sign search for a fixed subset R:
pruning (once |partial column sum| exceeds 1 + remaining unsigned nonzeros
in that column, no completion can recover — and at the leaves this invariant
IS the Ghouila-Houri bound, so no final scan is needed) and sign symmetry
(the first selected row takes +1 WLOG, halving the tree).

### GF(2) rank prefilter in the bipartition search
For integer matrices, rank over GF(2) never exceeds rank over ℚ (a k×k
submatrix nonsingular mod 2 has odd, hence nonzero, determinant). For
{-1,0,1} matrices the support pattern IS M mod 2, so `_decompose` computes a
word-parallel XOR-basis rank of row-support bitmasks (capped at 3) before
touching the Float64 elimination path; candidates whose GF(2) lower bounds
already exceed the rank budget — the overwhelming majority — are rejected in
a few dozen bit operations. This gave ~20× on the 12×12 worst case (51s →
2.4s).

The inner rank computation uses `_rank_int` (Bareiss integer elimination),
which is ~18× faster than `LinearAlgebra.rank` (SVD) for the small {-1,0,1}
matrices encountered here. The 2000-trial random test suite now completes
in ~30 seconds (previously ~150 CPU-minutes with SVD).

Further acceleration opportunities:
- Incremental rank updates: precompute column echelon form of SZ once,
  check each v for independence in O(m²) rather than recomputing from scratch
- Pruning the outer S,T loop using matroid intersection theory
- Caching rank computations for repeated column subsets

### Case 4 degeneracy handling
When the first rank(B) = rank(C) = 1 partition has a degenerate A or D,
`_decompose` is retried with `reject_degenerate_3sum = true`. If no
non-degenerate partition exists, or `_find_epsilon` finds no R–K path, the
matrix is decided with `_tu_partition`. No fuzz input has reached either
fallback from a non-degenerate start, so they are untested.

## Testing

### Oracle
`naive_is_totally_unimodular` checks all square submatrix determinants.
Exponential time but correct. Used to verify `is_totally_unimodular`.

### Random testing
- 2000 uniform random {-1,0,1} matrices of size 2-5 × 2-6 against the naive
  oracle, and 200 of size 5-8 × 5-10 against `_tu_partition`.
- An exception in `is_totally_unimodular` fails the test.

### Structured fuzz
Uniform random matrices almost never reach the decomposition cases. The
"structured fuzz" testset composes 1-/2-sums of F_1, F_2, K33, K33ᵀ and
random network matrices with random permutations, ±1 scalings and pivots,
flips one entry in about a third of them, and checks both routes against
`_tu_partition`. The bugs it has found occur at rates of 1/800 to 1/6000,
so the 2000 seeded inputs in the suite are a smoke test: after changing
`_apply_decomposition`, `_decompose` or the recursion, run the generator
over several seeds × ~8000 inputs. The generator caps matrices at 12×12, so
the `:decomposition` route never calls `_find_two_separation` on them; the
default route does. Raising the cap (18×18 was used when the 2-sum split
went in) exercises it on both.

### Known test matrices
- `F_1`, `F_2`: TU, non-network, non-decomposable
- `network_matrix`: 3×3 network matrix
- `M3`: 7×20 network matrix constructed by hand
- `non_network_tu`: TU but not a network matrix

## Reference Implementation

CMR (Combinatorial Matrix Recognition) is a C library implementing TU
recognition. Available at https://github.com/discopt/cmr under MIT license.
Could be used as an additional oracle for larger matrices.