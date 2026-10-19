# Implementation Notes

This document records key decisions, bugs found, and fixes applied during
development of the TotalUnimodularity.jl package. Intended as context for
future developers and AI assistants working on this codebase.

## Development History

The package was implemented incrementally following Schrijver Chapter 19-20,
with each component tested against `is_totally_unimodular(M; algorithm = :naive)` (formerly the naive test (`algorithm = :naive`)) as an oracle.

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
but in practice all tested matrices agree with the naive test (`algorithm = :naive`).

**Superseded (October 2026):** the check has been removed; see "Case 4
accepts degenerate A and D" below.

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

Cases 5/6 no longer search the pivoted matrix (see "Cases 5/6 carry the
split through the pivot" below), which removes the source of these cycles.
Every recursive call is now on a strictly smaller matrix (see "Termination"
under Known Limitations), so the guard and the `seen` set were removed in
October 2026, together with the pivot-depth cap.

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
the naive test (`algorithm = :naive`) on random {-1, 0, 1} matrices.

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
`_is_tu_irreducible` runs, in order:

1. the network, transpose-network and special-matrix tests;
2. `_find_two_separation` (see below) — on success the block is split with
   `_apply_decomposition` (Case 2) and both pieces recurse from the top;
3. the short-cycle pre-filter `_tu_short_cycles`, under a work budget of
   4·m·n·(m+n) steps. It finds any 2×2 or 3×3 violation and settles most
   non-TU inputs; without it, random non-TU matrices each cost a full
   separation search (45× slower in aggregate on a sample of random 30×30
   and 60×60 matrices). A smallest violation of size 2 or 3 is Eulerian
   with no zero row, so its support is a 4-cycle or a chordless 6-cycle,
   and the filter walks those cycles in the support graph. It used to be
   the Eulerian test `_tu_eulerian` cut off at k = 3, which enumerates row
   triples and scans every column for each, O(m³n) even on a sparse
   matrix: on a sparse 378×376 TU block it spent the whole budget, a
   quarter of the run, and found nothing. The two agree on 200,000 random
   matrices. On dense matrices both are O(m³n³), hence the budget, and
   hence the filter runs after the 2-sum split. A spent budget only means
   the search decides instead. Of the settings measured (no 3×3 pass, and
   budgets of ¼, 1 and 4 times m·n·(m+n)), the last was fastest overall;
4. for a block with min(m,n) ≤ 8 (`_PARTITION_MAX_DIM`), the exact
   branch-and-prune Ghouila-Houri test `_tu_partition`. Measured on TU blocks
   with no 2-separation, the search route below is level with it at 8 and
   ahead beyond: 4 ms against 6 ms at 15×13, 0.7 ms against 0.6 s at 21×19;
5. `_find_three_separation` (see below). No separation means not TU, by
   Seymour's theorem; otherwise `_apply_decomposition` handles the split
   (Case 4 directly, Cases 5/6 through a pivot).

Until October 2026 step 5 did not exist: blocks up to min-dimension 24 went
to `_tu_partition` (about 13 s at 22, exponential), and larger ones to the
matroid-intersection search of Theorem 20.2, O((m+n)^8), which never
finished on anything of that size.

The exhaustive ≤12×12 bipartition search in `_decompose` is used only with
`fast = false`, i.e. `is_totally_unimodular(M; algorithm = :exhaustive)`,
which also skips steps 2, 4 and 5 for those blocks. It is an independent
implementation of the search, and every oracle and regression test runs
through both routes. The matroid-intersection search (`_decompose_matroid`,
`_solve_submodular` and their rank caches) was removed once nothing could
reach it; `_decompose` now refuses matrices above 12×12.

### Termination, and the one remaining fallback
Every recursive call of `_is_tu_recursive` is on a matrix with strictly
fewer rows plus columns, so the recursion terminates and no matrix can come
back on a recursion path. Call the number of rows plus columns of a side its
element count. A 1-sum component is a proper part of the matrix. A 2-sum
piece has the element count of its side plus 1, and each side of a
2-separation has at least 2. A 3-sum piece (Case 4) has the element count of
its side plus 3, and each side has at least 4: `_find_three_separation`
takes closed sets of 2 to N − 6 of the N − 4 free nodes, plus 2 fixed
elements per side, and `_decompose` requires 4. Cases 5/6 do not recurse on
the pivoted matrix: they move the pivot row to one side and the pivot column
to the other, so both element counts stay the same, and go straight to
Case 4 on that split. `_reduce` only removes rows and columns.

Until October 2026 there were two more guards, both falling back to the
exponential `_tu_partition`: a `seen` set of the matrices on the recursion
path (a repeat meant no progress), and a cap of 100 nested Case 5/6 pivots.
By the argument above the first could never fire, and the second could only
fire on a chain of over 100 nested rank-2 3-sums, where it would have caused
the stall it was meant to prevent (measured pivot depth was at most 1, up to
378×376 and after 200 random pivots). Both were removed; hashing every
matrix into `seen` had cost nothing measurable.

`_tu_partition` remains as a fallback in one place: `_find_epsilon` finding
no R–K path in Case 4. It goes through `_fallback`, which counts it in
`_FALLBACKS` and emits a warning naming the reason and the block size (at
most 10 per session); the last testset of the suite checks that the count
did not move, and `benchmark/fuzz_cmr.jl` prints it. None of the fallbacks
has fired since the pivot change below: in October 2026, 125,000 fuzz inputs
were run with every fallback logged before the guards were removed, and
65,000 after.

It cannot happen on the default route. Suppose `_find_epsilon(A, R, K)`
finds no path from R (rows of A meeting B) to K (columns of A meeting C).
Let X be the rows and columns of A reachable from R. X misses K, and A has
no nonzero between X and the rest of A, so moving the rest of A to side 2
leaves a cross block of rank 1 on side 1 (the rows of B in X) and a zero one
on side 2 (C lives in the columns of K): a 2-separation, provided X has at
least 2 elements. If X is a single row r, then R = {r} and row r is zero in
A; moving r alone to side 2 makes B zero and leaves C rank 1, a 2-separation
with sides of at least 3 and 5. So no path implies a 2-separation of M. On
the default route Case 4 only sees blocks that `_find_two_separation` (which
is complete) found none in, or their pivots, which have the same
separations. On the `:exhaustive` route (blocks up to 12×12 only)
`_decompose` only looks for 2-separations with sides of at least 4, and a
reduced matrix can have one with a side of 3 (rows equal outside one column,
opposite in it), so there the argument does not rule the fallback out; at
that size it is cheap anyway. The fallback is kept on both routes rather
than made an error on the default one: if the argument were wrong, a slow
correct answer with a warning serves a user better than an exception.

### Precompilation workload
The first call in a session used to compile the whole decision procedure,
about 14 s. A PrecompileTools workload at the end of the module sends one
small matrix down each route (network and special tests, 1-, 2- and 3-sum
splits, the pivot cases, the Ghouila-Houri test, non-TU exits, and the
other public functions); the first call now takes about 10 ms, at the price
of about 20 s when the package is precompiled.

### 3-separation search (`_find_three_separation`)
The search extends the idea of the 2-separation search. Name the sides so
that B = M[R1, C2] contains a spanning-tree edge (i1, j1) of the support
graph. Two kinds of split with rank(B) + rank(C) = 2 remain:

- rank B = rank C = 1. Fix a nonzero (i2, j2) of C as well. A row on side 1
  pulls in the columns whose 2×2 minor with (i1, j1) is nonzero; a column on
  side 1 pulls in the rows whose 2×2 minor with (i2, j2) is nonzero.
- rank B = 2, C = 0. Fix (i2, j2) so that rows i1, i2 and columns j1, j2
  form a nonsingular block of B. A row on side 1 pulls in the columns whose
  3×3 minor with that block is nonzero; a column on side 1 pulls in the
  rows where it is nonzero.

(rank B = 0, rank C = 2 is the second kind with the sides renamed.) The four
fixed elements never appear in a rule, so a split is the two fixed elements
of side 1 plus a closed set of between 2 and m+n-6 of the other nodes.
`_rule_sccs!` computes strongly connected components, `_closed_set` picks a
union of them in range. Usually a prefix of Tarjan's order works; when none
does there are at most three components and all unions are tried — two
sinks with no edge between them form a closed set that is not a prefix.

When there is no separation — the case that ends in "not TU" — every pair
of fixed entries has to be tried, so the cost is (number of pairs) × (cost
of one pass). On a 65×62 matrix with no separation that was 5.0 s at first
and is 0.17 s now, through four changes:

- Candidates for (i2, j2) are cut down with a spanning forest, as described
  in the docstring: about m+n per (i1, j1) instead of every nonzero.
- The rules are written out as adjacency lists (`_RuleGraph`) before each
  pass instead of being evaluated as predicates inside Tarjan's algorithm.
  A rule can only fire where an entry is nonzero in the row or the fixed
  row, and in the column or the fixed column, so a list is built from two
  supports. The half of the rules that depends on (i1, j1) alone is built
  once per (i1, j1). For the second kind the Schur complement of (i1, j1)
  is formed once, which turns the 3×3 minors into 2×2 minors.
- The forest is built from the spanning tree's own edges first, and a
  candidate (i2, j2) that is a tree edge with an earlier turn as (i1, j1)
  is skipped. Exchanged, the two describe the same pair, so the earlier
  pass covers it — or defers to a still earlier one; the chain ends at a
  pass whose valid candidate is not earlier, which is tried. This halves
  the number of passes.
- No per-pass allocation in the common outcome (one component).

A later round (October 2026) took another 1.25× off this case and much more
off TU matrices:

- Most of the rule lists that depend on (i2, j2) do not: a column that is
  zero in row i2 points to the rows of its own support, and in the second
  kind a row that is zero in column j2 of S points to its support in S.
  These are written once (per search, or per (i1, j1)), and each pass
  overrides only the lists of the few rows or columns that differ
  (`_override!`, `_restore!`), where it used to rewrite all of them.
- `skip` and `onstack` are `Vector{Bool}` rather than `BitVector`; the
  bit extraction was a sixth of the pass.
- On a TU matrix the search stops at the first (i1, j1) that crosses the
  separation, and in breadth-first order the crossing tree edges come in
  runs: on the K_n ⊕₃ K_m* family, 5–10% of the tree, at positions 47–67
  of 549, say. `_spread` reorders the tree edges by a stride near L/φ, which
  brought the number of (i1, j1) tried at 378×376 from 33–73 to 2–6 on most
  instances (it can also be unlucky: 3 to 90 on one). A first version
  computed the stride inside the search, and the closure over a reassigned
  variable boxed it and made the whole loop type-unstable, 1.5× slower;
  hence the helper function.

The remaining cost is about (m+n)² passes of Tarjan's algorithm when there
is no separation. Measured with `benchmark/scaling.jl`, CMR is 1.6× to 2.7×
faster on such inputs up to 120×118 and 1.1× to 1.6× from 190×188 to
378×376 (18 s against 23 s at the largest): it enumerates 3-separation
candidates there too. On TU matrices of the family the two are now level
within a factor of about 2 either way (0.40 s against 0.76 s at 378×376,
unpivoted; 0.57 s against 0.41 s pivoted); before this round CMR was 8× to
12× faster from 276×274 up. CMR documents its regularity test as
O((m+n)⁵), a simplified version of Truemper's cubic algorithm.

Correctness matters more here than for the 2-separation search, because "no
separation" is reported as "not TU". The unit test compares against
brute-force enumeration of all splits on small matrices, and
`exhaustive = true` (every candidate for (i2, j2)) is kept as a second
implementation to compare with.

### Cases 5/6 carry the split through the pivot
A pivot on a nonzero of B exchanges the roles of its row and its column. The
same split of the matroid's elements, with the pivot row now on side 2 and
the pivot column on side 1, is a split of the pivoted matrix with rank(B) =
rank(C) = 1, so Case 4 applies to it at once (`_apply_pivoted`). The earlier
code reduced the pivoted matrix and searched it again; that search could
return another rank-2 split and pivot straight back, which is where the
pivot cycles of Bug 8 came from.

### Case 4 accepts degenerate A and D
Case 4 used to reject a split whose A or D had a row or column with at most
one nonzero, or a duplicate pair, and search for another split (see Bug 3).
Schrijver's construction needs M to be reduced, connected and free of
2-separations, not A and D. Bug 3 dates from before 2-separations were
split off first, when the matrix reaching Case 4 could still have one. On
larger matrices there is usually no split that passes the old test — none
on the K_n ⊕₃ K_m* family at any size — so the rejection was removed,
together with the `reject_degenerate_3sum` keyword of `_decompose` and the
`_is_degenerate` predicate.

### 2-separation search (`_find_two_separation`)
A 2-separation of a connected matrix is a split with one cross block zero
and the other of rank 1. Fix a nonzero entry (i0, j0) of the rank-1 block
M[R1, C2]; then "belongs to side 1" is closed under two single-premise
rules — a column on side 1 pulls in every row where it is nonzero (keeps
M[R2, C1] = 0), and a row r on side 1 pulls in every column c whose 2×2
minor on rows {i0, r}, columns {j0, c} is nonzero (keeps M[R1, C2] rank 1).
The rules are the edges of a digraph on the other m+n-2 rows and columns,
and a 2-separation with that pivot exists iff the digraph is not strongly
connected: any nonempty proper closed set is a valid side 1. One pass of
Tarjan's algorithm per pivot finds the strongly connected components,
successors first, so every prefix of that order ending at a component
boundary is a closed set. The rules are written as adjacency lists from the
row and column supports, with the same `_RuleGraph` and `_rule_sccs!` as
the 3-separation search; an earlier version evaluated them over dense rows
and columns, O(m·n) per pivot, which on a 3-connected 378×376 block (all
pivots tried, none succeeding) was half the run.

Only the edges of a spanning tree of the support graph are tried as pivots
(`_two_separation_pivots`): the two sides of a 2-separation of a connected
matrix are joined by some tree edge, and that edge lies in the rank-1 block
because the other cross block is zero. That is m+n-1 pivots instead of one
per nonzero, O((m+n) · m · n) in total.

Which split is returned matters as much as finding one. Peeling a small
piece off a long 2-sum chain leaves a remainder nearly as large as before,
and every level of the recursion pays for reduction, the network tests and
another search on it: a chain of k blocks then costs k passes over the whole
matrix. So for each pivot the prefix nearest to half of the rows and columns
is taken, and since a pivot inside a piece at the end of a chain can only
cut that piece off, up to seven pivots spread over the spanning tree are
compared first (stopping at a split with a quarter of the matrix on each
side). The 705×704 chain of 201 blocks went from 10 s to under 0.5 s with
this.

The function is specified for reduced, connected matrices, which is what
`_is_tu_irreducible` passes it; the unit test checks it against brute-force
enumeration of all splits under the same precondition.

### Duplicate rows and columns are found by hashing
`_reduce` drops all but the first of each class of equal-or-opposite rows
(and columns). This was a pairwise comparison, O(m²n), with an allocation
per pair for the negated row; `_reduce` runs at every level of the
recursion, and on the 705×704 chain it accounted for 91% of a 59 s run.
`_independent_mask` hashes each vector with the sign that makes its first
nonzero positive, chains vectors with equal hashes and compares those entry
by entry: linear in the size of the matrix, and no allocation per vector
(59 s → 10 s on that input, 133 ms → 42 ms on a 200×400 network matrix).

### Network test without building graphs
`_is_network_matrix` follows Theorem 20.1 step by step, and the first
version built each object the theorem names. Profiling a 200×400 network
matrix showed nearly all of the time going into that construction rather
than into deciding anything, so three steps now work on the matrix directly
(42 ms → 8 ms on that input; the answers were checked against the previous
implementation on 60,000 random matrices):

- `_find_disconnected_gi` built G_i for every row i with `Graphs.add_edge!`,
  one clique per column, to ask whether it is connected. Rows are adjacent
  in G_i when they share a column that is zero in row i, so a search that
  alternates between rows and such columns on precomputed supports gives
  the components in O(nnz) per i. The function returns
  `(i, components, orig)`; the graph is gone.
- Case 1 (all columns ≤ 2 nonzeros) built the row graph and called
  `Graphs.is_bipartite`. Bipartiteness of that graph is a 2-colouring
  constraint per column — different colours for a same-sign pair, equal for
  an opposite-sign pair — which a union-find with parities checks in one
  pass over the columns.
- The submatrices M_k kept all n columns of M, so the recursion carried the
  full width down to pieces with a handful of rows. Columns that are zero
  on the rows of M_k take no part in any step and are dropped
  (`_split_submatrices(...; drop_zero_columns = true)`).

`_build_h` returns adjacency lists and `_is_bipartite` two-colours them;
it runs once per level on the components, not once per row. With that,
Graphs.jl was no longer needed, and neither was Combinatorics.jl (the naive
checker steps through subsets with `_next_combination!`, the special-matrix
test uses a fixed list of the 120 permutations of 1:5). Dropping both took
`using TotalUnimodularity` from 0.8 s to 0.1 s.

### Special-matrix test derives the column permutation
`_is_sign_and_permutation_equivalent` tried every pair of row and column
permutations, 14,400, with a support check on each; on a matrix with F_2's
row-sum profile that is not F_2 under the first few row orders, that was
about 0.5 ms (the README's "R10" row, which is F_2 with rows permuted). Once
the rows are permuted, column j can only go to a column of the target with
the same support, so the column permutation is built from supports, with
backtracking only among columns of equal support, and the sign check runs on
each complete match: about 12 µs. It agrees with brute force (all
permutation pairs and all 2¹⁰ sign vectors) on 6,000 perturbed copies of
F_1 and F_2.

### No recursion depth limit
`depth` used to be incremented on every recursive call, with a cutoff at
100 that fell back to `_tu_partition`. A 2-sum chain of k blocks recurses
k deep, so a chain of more than 100 blocks hit the cutoff and handed a huge
matrix to the exponential test. It was then restricted to the pivot cases
(5/6), and in October 2026 removed altogether (see "Termination" above).

### Eulerian test skips zero columns
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

The two recursions are top-level functions (`_partition_subsets`,
`_partition_search`) sharing a `_PartitionState`, and `_tu_eulerian`'s are
likewise (`_euler_rows`, `_euler_cols`, `_EulerState`). As nested closures
that call themselves they were boxed (`Core.Box`), so every recursive call
was dynamically dispatched: lifting them made the Ghouila-Houri test 2.2×
and the Eulerian test 1.5× faster on small TU blocks (October 2026, found by
a review). The same pass removed boxing from `_independent_mask` (a
reassigned loop variable captured by `all(k -> …)`) and from Cases 5/6 of
`_apply_decomposition` (pivot indices assigned in both cases and captured by
comprehensions). Only the recursive `assign` in the 5×5 special-matrix test
is still boxed; it takes 34 µs.

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

## Testing

### Oracle
the naive test (`algorithm = :naive`) checks all square submatrix determinants.
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
the `:exhaustive` route never calls `_find_two_separation` on them; the
default route does. Raising the cap (18×18 was used when the 2-sum split
went in) exercises it on both.

For the 3-separation route there are two more generators, both built from
the K_n ⊕₃ K_m* family (3-sum of the network matrix of K_n with the
transpose of the network matrix of K_m plus a vertex of degree 3) by random
pivots, row and column deletions and entry flips. Small members are checked
against `_tu_partition` with `_PARTITION_MAX_DIM[] = 0`, which forces the
search route at every size (1500 inputs in the suite). Members from 21×19
to 66×64 can only be checked against the CMR binary:
`benchmark/fuzz_cmr.jl`, not in the suite because it needs that binary. CMR
with default options aborts on a few non-TU inputs (reported upstream,
October 2026; one such matrix is in the suite as `cmr_abort`), so the
script repeats a failed run with `--no-simple-3-sepa`.

### Review tests (`test/extra_tests.jl`)
Added from a pre-registration review (October 2026). Every expected answer
comes from theory rather than from running the package: incidence matrices
of digraphs and bipartite graphs and interval matrices (TU), odd-cycle
incidence matrices up to C₁₀₁ and odd cycles hidden in connected bipartite
graphs of up to 240 vertices (minimally non-TU, so from C₅ on the short-cycle
pre-filter cannot see them), degenerate shapes on all five algorithms, input
types (sparse, UInt8, BigInt, Bool, views, Float64, Rational, NaN),
metamorphic variants that preserve TU-ness either way, a threaded run, and
Aqua. CI runs the suite with 4 threads.

The largest odd-edge graph (240×360, not TU) takes 2–11 s, where CMR takes
4 ms: it passes the pre-filter, and the 3-separation search then tries every
pivot before concluding there is no separation. CMR's `--stats` show it
deciding at its (co)network stage with one Camion sign check, presumably by
recognising the support as graphic and then testing only the signs; a
support-graphicness test here would settle such matrices the same way.

### Known test matrices
- `F_1`, `F_2`: TU, non-network, non-decomposable
- `network_matrix`: 3×3 network matrix
- `M3`: 7×20 network matrix constructed by hand
- `non_network_tu`: TU but not a network matrix

## Reference Implementation

CMR (Combinatorial Matrix Recognition) is a C library implementing TU
recognition. Available at https://github.com/discopt/cmr under MIT license.
Could be used as an additional oracle for larger matrices.