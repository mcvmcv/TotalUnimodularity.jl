# Mathematical Background: Total Unimodularity

## Definition

A matrix M with integer entries is **totally unimodular (TU)** if every square
submatrix has determinant in {-1, 0, 1}.

A necessary condition is that all entries of M are in {-1, 0, 1}.

## Seymour's Decomposition Theorem (Corollary 19.6b, Schrijver)

Let M be a totally unimodular matrix. Then at least one of the following holds:

1. M or its transpose is a network matrix
2. M is one of the matrices F_1 or F_2 (possibly after permuting rows/columns
   or multiplying rows/columns by -1)
3. M has a row or column with at most one nonzero, or M has two linearly
   dependent rows or columns
4. The rows and columns of M can be permuted so that M = [A B; C D] with
   rank(B) + rank(C) ≤ 2, where both A and D have r + c ≥ 4

## Special Matrices (Section 19.4, equation 42)

F_1 and F_2 are the two 5×5 TU matrices that cannot be decomposed further:

    F_1 = [ 1 -1  0  0 -1]      F_2 = [1 1 1 1 1]
           [-1  1 -1  0  0]             [1 1 1 0 0]
           [ 0 -1  1 -1  0]             [1 0 1 1 0]
           [ 0  0 -1  1 -1]             [1 0 0 1 1]
           [-1  0  0 -1  1]             [1 1 0 0 1]

Two matrices are **equivalent** (for the purpose of this test) if one can be
obtained from the other by permuting rows/columns and multiplying rows/columns
by -1.

## Network Matrices (Example 4, Section 19.3)

Let D = (V, A) be a directed graph and T = (V, A_T) a directed tree on V.
The **network matrix** M is the A_T × A matrix defined by:

    M[a', a] = +1 if the unique path in T between endpoints of a traverses a' forwardly
             = -1 if the unique path traverses a' backwardly
             =  0 if the unique path does not traverse a'

Network matrices are closed under taking submatrices.

### Recognition Algorithm (Theorem 20.1)

**Case 1: All columns have ≤ 2 nonzeros**

Build undirected graph G on rows:
- Connect rows i and j with an edge if some column has nonzeros of the SAME
  sign in both rows
- Connect rows i and j via a path of length 2 (new intermediate vertex) if
  some column has nonzeros of OPPOSITE sign in both rows

M is a network matrix iff G is bipartite.

**Case 2: Some column has ≥ 3 nonzeros**

For each row index i, build graph G_i on rows {1,...,m}\{i}:
- j and k are adjacent iff some column has nonzeros in rows j and k but zero
  in row i

If all G_i are connected → M is NOT a network matrix.

If some G_i is disconnected (say G_1 with components C_1,...,C_p):

Define:
- W = support of row 1 (column indices where row 1 is nonzero)
- W_i = W ∩ support of row i (for i ≥ 2)
- U_k = ∪{W_i | i ∈ C_k}

Build graph H on components C_1,...,C_p where C_k and C_l are adjacent iff:
- ∃ i ∈ C_k : U_l ⊄ W_i and U_l ∩ W_i ≠ ∅, AND
- ∃ j ∈ C_l : U_k ⊄ W_j and U_k ∩ W_j ≠ ∅

Let M_k = submatrix of M consisting of row 1 and rows indexed by C_k.

**M is a network matrix iff H is bipartite and each M_k is a network matrix.**

## Seymour Decomposition Operations

### 1-sum
Given matrices A (r_A × c_A) and B (r_B × c_B), their 1-sum is the block
diagonal matrix:

    [A  0]
    [0  B]

### 2-sum
Given A with distinguished last column a, and B with distinguished first row b^T,
their 2-sum is:

    [A_m    a⊗b]
    [0      B_m]

where A_m = A without last column, B_m = B without first row.

### 3-sum
Given A of the form [A_m  a  a; c^T  0  1] and B of the form [1  0  b^T; d  d  B_m],
their 3-sum is:

    [A_m    a⊗b^T]
    [d⊗c^T  B_m  ]

If A and B are both TU, so are their 1-sum, 2-sum, and 3-sum.

## The TU Test Algorithm (Theorem 20.3, Schrijver)

### Preprocessing
1. Check all entries are in {-1, 0, 1}
2. Repeatedly delete rows/columns with ≤ 1 nonzero
3. Repeatedly delete one of each pair of linearly dependent rows/columns
   (i.e. pairs where one row/column equals ±1 times another)
4. Repeat until stable

### Main Algorithm
After preprocessing, test in order:

1. Is M a network matrix? (Theorem 20.1)
2. Is M^T a network matrix?
3. Is M equivalent to F_1 or F_2?
4. Try Seymour decomposition (Theorem 20.2): find partition M = [A B; C D]
   with rank(B) + rank(C) ≤ 2 and r+c ≥ 4 for both A and D
   - If no decomposition exists → NOT TU
   - If decomposition found → recurse on Cases 1-6:

**Case 1:** rank(B) = rank(C) = 0
M is TU iff A and D are TU.

**Case 2:** rank(B) = 1, rank(C) = 0
Write B = f⊗g (f is {0,±1} column, g is {0,+1} row).
M is TU iff [A f] and [g; D] are TU.

**Case 3:** rank(B) = 0, rank(C) = 1
Symmetric to Case 2.
Write C = f⊗g.
M is TU iff [A; g] and [f D] are TU.

**Case 4:** rank(B) = rank(C) = 1
Requires A and D to be non-degenerate (no trivial/dependent rows or cols).

Write B = f_B⊗g_B and C = f_C⊗g_C.

Normalise:
- B_rows = rows where f_B ≠ 0
- C_cols = cols where g_C ≠ 0
- Scale rows of A in B_rows by f_B[i] to make B = [0; 1_block]
- Scale rows of D in C_rows by f_C[i] to make C = [1_block 0; 0]

This puts M in the standard form (28):

    M = [A1  A2   0   0]
        [A3  A4   1   0]   ← B_rows
        [0    1  D1  D2]   ← C_rows of D
        [0    0  D3  D4]

where A = [A1 A2; A3 A4] and D = [D1 D2; D3 D4].

Find ε₁ ∈ {+1,-1} from A:
- Build bipartite graph G on rows and columns of A
- R = rows intersecting A4, K = columns intersecting A4
- If A4 has a nonzero entry, ε₁ = that entry
- Otherwise find shortest path Π from R to K in G
  δ = sum of A entries on edges of Π (odd length path, so δ is odd)
  ε₁ = +1 if δ ≡ 1 (mod 4), -1 if δ ≡ -1 (mod 4)

Find ε₂ similarly from D (using C_rows as R, B_cols as K).

M is TU iff both of these matrices are TU:

    mat1 = [A1        A2        0_{nnotR×1}  0_{nnotR×1}]
           [A3        A4        1_{nR×1}     1_{nR×1}   ]
           [0_{1×nnotK} 1_{1×nK}  0            ε₂        ]

    mat2 = [ε₁         0_{1×nBK}      1_{1×nnotBK}    0        ]
           [1_{nCR×1}  1_{nCR×1}      D1              D2       ]
           [0_{nnotCR} 0_{nnotCR}     D3              D4       ]

where nR = |B_rows|, nK = |C_cols|, nnotR = |notB_rows|, nnotK = |notC_cols|,
nCR = |C_rows|, nBK = |B_cols|, nnotCR = |notC_rows|, nnotBK = |notB_cols|.

**Case 5:** rank(B) = 2, rank(C) = 0
Pivot on a nonzero entry of B to reduce to Case 4.
Find first nonzero B[i,j] = η. Permute M so this entry is at position (1,1)
and pivot on the leading 1×1 submatrix. With the pivot row moved to the
bottom part and the pivot column to the left part, the pivoted matrix has a
partition with rank(B) = rank(C) = 1, and Case 4 is applied to it.

**Case 6:** rank(B) = 0, rank(C) = 2
Symmetric to Case 5, pivot on a nonzero entry of C.

### Cycle Detection
If the pivoted matrix of Cases 5 and 6 is searched for a decomposition
afresh, the search can return another Case 5/6 partition and pivot straight
back. The implementation avoids this by carrying the partition through the
pivot (see "3-separations in polynomial time" below), so the pivoted matrix
goes to Case 4 directly. As a safeguard the matrices on the current
recursion path are still recorded; if one is encountered again it is decided
with the exact Ghouila-Houri test. A cycle says nothing about total
unimodularity.

### 2-separations in polynomial time
Cases 2 and 3 (rank(B) + rank(C) = 1) do not need the general search of
Theorem 20.2. Suppose M is connected, C = 0 and rank(B) = 1, and fix a
nonzero entry (i0, j0) of B, so row i0 is in the top part and column j0 in
the right part. Then:

- if a column is in the left part, every row where it is nonzero must be in
  the top part (otherwise C ≠ 0);
- if a row r is in the top part, every column c with
  M[i0,j0]·M[r,c] ≠ M[i0,c]·M[r,j0] must be in the left part (otherwise B has
  a nonzero 2×2 minor through (i0, j0), and a matrix with a nonzero entry has
  rank 1 iff all such minors vanish).

Each rule has a single premise, so together they define a digraph on the
remaining rows and columns, and the admissible top-left parts are exactly
the nonempty proper subsets closed under its edges. One exists iff the
digraph is not strongly connected. Trying every nonzero entry as (i0, j0)
finds a 2-separation whenever there is one; the case rank(C) = 1, B = 0 is
the same statement with the two sides renamed.

It is enough to try the edges of a spanning tree of the support graph of M
(rows and columns as vertices, nonzero entries as edges): the tree connects
the two sides of the separation, so one of its edges crosses, and since
C = 0 that edge is a nonzero of B.

### 3-separations in polynomial time
The same idea finds the partitions of Cases 4–6 (rank(B) + rank(C) = 2, at
least four rows-plus-columns on each side) when M is connected and has no
2-separation. Name the sides so that B contains an edge (i1, j1) of a
spanning tree of the support graph.

*rank(B) = rank(C) = 1.* Fix a nonzero (i2, j2) of C. With p = M[i1,j1] and
q = M[i2,j2]:

- a row r in the top part forces every column c with
  p·M[r,c] ≠ M[r,j1]·M[i1,c] into the left part (rank(B) = 1);
- a column c in the left part forces every row r with
  q·M[r,c] ≠ M[r,j2]·M[i2,c] into the top part (rank(C) = 1).

*rank(B) = 2, C = 0.* Fix (i2, j2) so that rows i1, i2 and columns j1, j2
form a nonsingular 2×2 block X of B; for a rank-2 block and any nonzero
(i1, j1) of it one exists.

- a row r in the top part forces every column c for which the 3×3 minor on
  rows {i1, i2, r} and columns {j1, j2, c} is nonzero into the left part (a
  matrix containing a nonsingular 2×2 block has rank 2 iff every 3×3 minor
  containing the block vanishes);
- a column c in the left part forces every row where it is nonzero into the
  top part (C = 0).

In both cases the four fixed rows and columns never occur in a rule (the
minors vanish identically for them), so the admissible top-left parts are
the fixed elements of that side together with a set of the remaining rows
and columns that is closed under the rules and leaves at least two of them
out and takes at least two in.

After a pivot on (i1, j1) in the second case, the partition with row i1
moved to the bottom and column j1 moved to the left is of the first kind,
which is how Cases 5 and 6 are reduced to Case 4.

## Seymour Decomposition Test (Theorem 20.2)

Find Y ⊆ columns of [I | M] such that:
- ρ(Y) + ρ(X\Y) ≤ ρ(X) + 2
- |Y| ≥ 4, |X\Y| ≥ 4
- Y intersects both the I columns and the M columns
- X\Y intersects both the I columns and the M columns

Schrijver solves this by iterating over all S, T ⊆ X with |S| = |T| = 4
(satisfying the intersection conditions) and minimising the submodular
function ρ(Y) + ρ(X\Y) subject to S ⊆ Y ⊆ X\T. Y∩XI gives the top row
partition and Y∩XM the left column partition, which determine A, B, C, D.

That is polynomial but O(|X|^8) in the outer loop alone, |X| = m + n. An
implementation of it was part of this package until October 2026 and never
finished on a matrix larger than about 12×12, so it was removed. The
partition is found with the two searches described above instead; for
matrices up to 12×12 there is also an exhaustive enumeration of all
partitions (`_decompose`), used as an independent check.

## Known Issues and Limitations

1. **Performance:** the general separation search of Theorem 20.2 is
   impractical and is not implemented; 2- and 3-separations are found by
   the searches above. The 3-separation search is O((m+n)² · m · n) when there
   is no separation, well above the cubic bound of Truemper's algorithm.
   See IMPLEMENTATION_NOTES.md for the routing and measurements.

2. **Exponential fallback:** the exact Ghouila-Houri test decides blocks of
   smaller dimension at most 8 and is the fallback when a matrix repeats on
   the recursion path or the sign ε of a 3-sum cannot be determined. No
   test input reaches the fallback on the current code.

## References

- Schrijver, A. (1986). *Theory of Linear and Integer Programming*.
  Wiley. Chapter 19-20.
- Seymour, P.D. (1980). Decomposition of regular matroids.
  *Journal of Combinatorial Theory, Series B*, 28(3), 305-359.
- Ghouila-Houri, A. (1962). Caractérisation des matrices totalement
  unimodulaires. *Comptes Rendus de l'Académie des Sciences*, 254, 1192-1194.