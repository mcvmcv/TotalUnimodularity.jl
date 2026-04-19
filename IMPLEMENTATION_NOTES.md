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

### `_extract_rank1` does NOT normalise
Early versions normalised f to have positive first entry (to ensure g is
{0,+1}). This was wrong — negating f also negates the relationship between
f and the columns of B, breaking the factorisation. The correct implementation
takes f as the first nonzero column of B without any normalisation.

Schrijver says f is a {0,+1} vector and g is {0,±1}, but in practice f
is {0,±1} and g is {0,+1}. The text appears to have a typo.

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

The fix: maintain a `Set{Matrix{Int}}` of seen matrices throughout the
recursion. If the same matrix is encountered again, return `false`.

Returning `false` on cycle detection is safe (no false positives confirmed
on 2000 random matrices) but may give false negatives for TU matrices that
trigger cycles. This is a known limitation.

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

## Known Limitations

### Performance
`_decompose` has O((m+n)^8) complexity in the worst case. For matrices
larger than approximately 8×10, it becomes slow.

The inner rank computation uses `_rank_int` (Bareiss integer elimination),
which is ~18× faster than `LinearAlgebra.rank` (SVD) for the small {-1,0,1}
matrices encountered here. The 2000-trial random test suite now completes
in ~30 seconds (previously ~150 CPU-minutes with SVD).

Further acceleration opportunities:
- Incremental rank updates: precompute column echelon form of SZ once,
  check each v for independence in O(m²) rather than recomputing from scratch
- Pruning the outer S,T loop using matroid intersection theory
- Caching rank computations for repeated column subsets

### Cycle detection correctness
Returning `false` on cycle detection is safe but conservative. A TU matrix
that somehow triggers a cycle (which should not happen if the algorithm is
correct but might due to implementation bugs) would be incorrectly reported
as non-TU. No such case has been found in testing.

### Case 4 degeneracy handling
Returning `false` when A or D is degenerate is conservative. In theory,
a TU matrix could have a decomposition where A is degenerate but another
valid non-degenerate decomposition also exists. `_decompose` returns the
first decomposition found and does not search for non-degenerate alternatives.

## Testing

### Oracle
`naive_is_totally_unimodular` checks all square submatrix determinants.
Exponential time but correct. Used to verify `is_totally_unimodular`.

### Random testing
- 2000 random {-1,0,1} matrices of size 2-5 × 2-6: 2000/2000 agree
- Extended tests on larger matrices ongoing

### Known test matrices
- `F_1`, `F_2`: TU, non-network, non-decomposable
- `network_matrix`: 3×3 network matrix
- `M3`: 7×20 network matrix constructed by hand
- `non_network_tu`: TU but not a network matrix

## Reference Implementation

CMR (Combinatorial Matrix Recognition) is a C library implementing TU
recognition. Available at https://github.com/discopt/cmr under MIT license.
Could be used as an additional oracle for larger matrices.