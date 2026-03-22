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
Find first nonzero B[i,j] = η. Permute M so this entry is at position (1,1),
pivot on the leading 1×1 submatrix, reduce, and recurse.

**Case 6:** rank(B) = 0, rank(C) = 2
Symmetric to Case 5, pivot on a nonzero entry of C.

### Cycle Detection
Cases 5 and 6 can cycle (the pivot may produce a matrix that decomposes
again as Case 5/6, leading to an infinite loop). A set of previously seen
matrices is maintained; if the same matrix is encountered again, return false.

## Seymour Decomposition Test (Theorem 20.2)

Find Y ⊆ columns of [I | M] such that:
- ρ(Y) + ρ(X\Y) ≤ ρ(X) + 2
- |Y| ≥ 4, |X\Y| ≥ 4
- Y intersects both the I columns and the M columns
- X\Y intersects both the I columns and the M columns

This is solved by iterating over all S, T ⊆ X with |S| = |T| = 4 (satisfying
the intersection conditions) and solving the submodular minimisation problem:

    min ρ(Y) + ρ(X\Y) subject to S ⊆ Y ⊆ X\T

The submodular minimisation uses a BFS-based path augmentation algorithm
(see _solve_submodular in the source).

Y∩XI gives the top row partition, Y∩XM gives the left column partition,
which determines A, B, C, D.

**Performance note:** The outer loop is O(|X|^8) where |X| = m + n.
For a 10×15 matrix this is O(25^8) ≈ 1.5 × 10^11 — very slow.
Performance optimisation of this step is a significant open task.

## Known Issues and Limitations

1. **Performance:** `_decompose` (Theorem 20.2) is very slow for matrices
   larger than ~6×6. The O(N^8) outer loop needs optimisation.

2. **Cycle detection:** Cases 5 and 6 use a `seen` set to detect cycles.
   Returning `false` on cycle detection is safe (no false positives) but
   may give false negatives for some TU matrices if the algorithm cycles.
   In practice, all tested matrices agree with `naive_is_totally_unimodular`.

3. **Case 4 degeneracy:** When the decomposition produces A or D with
   dependent/trivial rows or columns, Case 4 returns `false` defensively.
   This is correct for non-TU matrices but may give false negatives for
   some TU matrices.

## References

- Schrijver, A. (1986). *Theory of Linear and Integer Programming*.
  Wiley. Chapter 19-20.
- Seymour, P.D. (1980). Decomposition of regular matroids.
  *Journal of Combinatorial Theory, Series B*, 28(3), 305-359.
- Ghouila-Houri, A. (1962). Caractérisation des matrices totalement
  unimodulaires. *Comptes Rendus de l'Académie des Sciences*, 254, 1192-1194.