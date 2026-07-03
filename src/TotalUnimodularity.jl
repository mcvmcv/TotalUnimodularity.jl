module TotalUnimodularity

using LinearAlgebra
using Combinatorics
using Graphs

# Public API
export naive_is_totally_unimodular
export is_totally_unimodular
export cmr_is_totally_unimodular
export one_sum, two_sum, three_sum
export pivot
export F_1, F_2

# ──────────────────────────────────────────────────────────────────────────────
# Special matrices (Seymour's theorem)
# ──────────────────────────────────────────────────────────────────────────────

"""
    F_1

The first special totally unimodular matrix in Seymour's decomposition theorem.
This 5×5 matrix is TU but cannot be decomposed via 1-, 2-, or 3-sums from
smaller TU matrices.
"""
const F_1 = [ 1 -1  0  0 -1
             -1  1 -1  0  0
              0 -1  1 -1  0
              0  0 -1  1 -1
             -1  0  0 -1  1]

"""
    F_2

The second special totally unimodular matrix in Seymour's decomposition theorem.
This 5×5 matrix is TU but cannot be decomposed via 1-, 2-, or 3-sums from
smaller TU matrices.
"""
const F_2 = [1 1 1 1 1
             1 1 1 0 0
             1 0 1 1 0
             1 0 0 1 1
             1 1 0 0 1]

# ──────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ──────────────────────────────────────────────────────────────────────────────

# Check that a matrix has r + c ≥ 4. Returns (r, c) if valid.
function _check_size(A::Matrix{Int})
    r, c = size(A)
    r + c < 4 && error("The number of rows plus columns of each matrix must be at least four.")
    return r, c
end

# Return true if v is a standard basis vector or the zero vector.
_is_trivial_vector(v::AbstractVector) = count(!iszero, v) <= 1

# Drop all slices along `dim` that are trivial (zero or standard basis vectors).
# dim=1 drops trivial rows; dim=2 drops trivial columns.
function _drop_trivial_vectors(M::Matrix{Int}, dim::Int)
    mask = [!_is_trivial_vector(s) for s in eachslice(M, dims=dim)]
    return dim == 1 ? M[mask, :] : M[:, mask]
end

# Repeatedly drop trivial rows and columns until the matrix stabilises.
function _reduce_trivial_vectors(M::Matrix{Int})
    while true
        N = _drop_trivial_vectors(_drop_trivial_vectors(M, 1), 2)
        N == M && return M
        M = N
    end
end

# Check if matrix M is equivalent to target under ±1 row/column scalings
# and row/column permutations. Assumes M and target are both 5×5 {-1,0,1} matrices.
function _is_sign_and_permutation_equivalent(M::Matrix{Int}, target::Matrix{Int})
    n = 5
    # Pre-allocate buffers — reused across all iterations
    row_signs = zeros(Int, n)
    col_signs = zeros(Int, n)
    queue = Vector{Tuple{Int,Int}}(undef, 2n)

    for row_perm in permutations(1:n)
        for col_perm in permutations(1:n)

            # Cheap sparsity check before doing any sign work
            sparsity_ok = true
            for i in 1:n
                for j in 1:n
                    if iszero(M[i,j]) != iszero(target[row_perm[i], col_perm[j]])
                        sparsity_ok = false
                        @goto next_col_perm
                    end
                end
            end

            # BFS sign propagation
            fill!(row_signs, 0)
            fill!(col_signs, 0)
            row_signs[1] = 1
            queue[1] = (0, 1)  # 0 = row, 1 = col
            queue_head = 1
            queue_tail = 1

            while queue_head <= queue_tail
                (dim, idx) = queue[queue_head]
                queue_head += 1

                if dim == 0  # row
                    for j in 1:n
                        M[idx, j] == 0 && continue
                        required = target[row_perm[idx], col_perm[j]] * row_signs[idx] * M[idx, j]
                        if col_signs[j] == 0
                            col_signs[j] = required
                            queue_tail += 1
                            queue[queue_tail] = (1, j)
                        elseif col_signs[j] != required
                            @goto next_col_perm
                        end
                    end
                else  # col
                    for i in 1:n
                        M[i, idx] == 0 && continue
                        required = target[row_perm[i], col_perm[idx]] * col_signs[idx] * M[i, idx]
                        if row_signs[i] == 0
                            row_signs[i] = required
                            queue_tail += 1
                            queue[queue_tail] = (0, i)
                        elseif row_signs[i] != required
                            @goto next_col_perm
                        end
                    end
                end
            end

            return true  # consistent sign assignment found

            @label next_col_perm
        end
    end
    return false
end

"""
    _is_special_matrix(M)

Test whether `M` is equivalent to [`F_1`](@ref) or [`F_2`](@ref) under
row/column permutations and ±1 row/column scalings.
"""
function _is_special_matrix(M::Matrix{Int})
    size(M) == (5, 5) || return false
    # The multiset of absolute row sums is invariant under sign/permutation equivalence.
    # F_1 has profile [3,3,3,3,3]; F_2 has profile [3,3,3,3,5].
    # This O(n) check rejects most non-equivalent matrices before the O(14400) loop.
    row_abs_sums = sort(ntuple(i -> sum(abs, @view M[i,:]), 5))
    if row_abs_sums == (3,3,3,3,3)
        _is_sign_and_permutation_equivalent(M, F_1) && return true
    end
    if row_abs_sums == (3,3,3,3,5)
        _is_sign_and_permutation_equivalent(M, F_2) && return true
    end
    return false
end

# Return true if any two rows of M are equal or negatives of each other.
function _has_dependent_rows(M::Matrix{Int})
    r = size(M, 1)
    for i in 1:r-1
        for j in i+1:r
            @views M[i,:] == M[j,:] && return true
            @views M[i,:] == -M[j,:] && return true
        end
    end
    return false
end

# Return true if any two columns of M are equal or negatives of each other.
function _has_dependent_cols(M::Matrix{Int})
    c = size(M, 2)
    for i in 1:c-1
        for j in i+1:c
            @views M[:,i] == M[:,j] && return true
            @views M[:,i] == -M[:,j] && return true
        end
    end
    return false
end

# Return true if M has any dependent rows or columns.
_has_dependent_vectors(M::Matrix{Int}) =
    _has_dependent_rows(M) || _has_dependent_cols(M)

# Return true if M is "degenerate" for the 3-sum (Case 4) construction:
# any row/column has ≤1 nonzero, or any two rows/columns are equal or opposite.
_is_degenerate(M::Matrix{Int}) =
    any(i -> count(!iszero, @view M[i,:]) <= 1, 1:size(M,1)) ||
    any(j -> count(!iszero, @view M[:,j]) <= 1, 1:size(M,2)) ||
    _has_dependent_vectors(M)

# Remove one row from each dependent pair of rows.
function _drop_dependent_rows(M::Matrix{Int})
    r = size(M, 1)
    keep = trues(r)
    for i in 1:r-1
        keep[i] || continue
        for j in i+1:r
            if @views M[i,:] == M[j,:] || M[i,:] == -M[j,:]
                keep[j] = false
            end
        end
    end
    return M[keep, :]
end

# Remove one column from each dependent pair of columns.
function _drop_dependent_cols(M::Matrix{Int})
    c = size(M, 2)
    keep = trues(c)
    for i in 1:c-1
        keep[i] || continue
        for j in i+1:c
            if @views M[:,i] == M[:,j] || M[:,i] == -M[:,j]
                keep[j] = false
            end
        end
    end
    return M[:, keep]
end

# Remove dependent rows and columns.
_drop_dependent_vectors(M::Matrix{Int}) =
    _drop_dependent_cols(_drop_dependent_rows(M))

# Integer rank via Bareiss elimination — in-place, destroys B.
function _rank_int!(B::Matrix{Int})::Int
    m, n = size(B)
    (m == 0 || n == 0) && return 0
    prev = 1
    r = 0
    for col in 1:n
        prow = 0
        for row in r+1:m
            iszero(B[row, col]) || (prow = row; break)
        end
        prow == 0 && continue
        r += 1
        if prow != r
            for c in 1:n; B[r,c], B[prow,c] = B[prow,c], B[r,c]; end
        end
        for row in r+1:m
            factor = B[row, col]
            for c in col+1:n
                B[row, c] = (B[r, col] * B[row, c] - factor * B[r, c]) ÷ prev
            end
            B[row, col] = 0
        end
        prev = B[r, col]
        r == m && break
    end
    r
end

# Integer rank via Bareiss elimination — exact, no BLAS, no Float64 conversion.
# Significantly faster than LinearAlgebra.rank for small {-1,0,1} matrices.
_rank_int(A::AbstractMatrix{Int})::Int = (size(A,1)==0 || size(A,2)==0) ? 0 : _rank_int!(Matrix{Int}(A))

# Exact integer determinant via Bareiss elimination — in-place, destroys B.
# The final pivot equals det(B); row swaps flip the sign. Intermediate values
# are minors of B (Hadamard bound n^(n/2)), so no Int64 overflow for the
# {-1,0,1} matrices up to ~15×15 that naive_is_totally_unimodular can handle.
function _det_int!(B::Matrix{Int})::Int
    n = size(B, 1)
    n == 0 && return 1
    sign = 1
    prev = 1
    for k in 1:n-1
        if iszero(B[k, k])
            prow = 0
            for row in k+1:n
                iszero(B[row, k]) || (prow = row; break)
            end
            prow == 0 && return 0
            for c in 1:n; B[k,c], B[prow,c] = B[prow,c], B[k,c]; end
            sign = -sign
        end
        for row in k+1:n
            for c in k+1:n
                B[row, c] = (B[k, k] * B[row, c] - B[row, k] * B[k, c]) ÷ prev
            end
            B[row, k] = 0
        end
        prev = B[k, k]
    end
    sign * B[n, n]
end

# Float64 Gaussian elimination with partial pivoting, in-place, destroys B.
# Correct for {-1,0,1} matrices with n ≤ ~20 rows/cols: with partial pivoting,
# entries stay ≤ 2^(n-1) in magnitude and legitimate nonzeros are rationals no
# smaller than 2^-(n-1) ≈ 2e-6, while accumulated rounding noise on should-be-
# zero entries stays below ~n·2^n·eps ≈ 5e-9. The 1e-7 pivot threshold sits
# safely between the two. Integer division (the slow step in Bareiss) is
# replaced by FP multiply-subtract, giving a 3-5x speedup in practice.
function _rank_float!(B::Matrix{Float64})::Int
    m, n = size(B)
    r = 0
    @inbounds for col in 1:n
        prow = 0
        best = 1e-7  # see threshold note above
        for row in r+1:m
            v = abs(B[row, col])
            if v > best; best = v; prow = row; end
        end
        prow == 0 && continue
        r += 1
        if prow != r
            for c in 1:n; B[r,c], B[prow,c] = B[prow,c], B[r,c]; end
        end
        pivot = B[r, col]
        for row in r+1:m
            factor = B[row, col] / pivot
            for c in col+1:n
                B[row, c] -= factor * B[r, c]
            end
            B[row, col] = 0.0
        end
        r == m && break
    end
    r
end

# Gaussian elimination on B[1..m, 1..n] — works on the leading m×n block of a
# pre-allocated scratch buffer, so no heap allocation.
function _rank_float_view!(B::Matrix{Float64}, m::Int, n::Int)::Int
    r = 0
    @inbounds for col in 1:n
        prow = 0; best = 1e-7  # see threshold note on _rank_float!
        for row in r+1:m
            v = abs(B[row, col])
            if v > best; best = v; prow = row; end
        end
        prow == 0 && continue
        r += 1
        if prow != r
            for c in 1:n; B[r,c], B[prow,c] = B[prow,c], B[r,c]; end
        end
        piv = B[r, col]
        for row in r+1:m
            fac = B[row, col] / piv
            for c in col+1:n; B[row,c] -= fac * B[r,c]; end
            B[row, col] = 0.0
        end
        r == m && break
    end
    r
end

# GF(2) rank of the support-pattern rows {srow[i] & colmask : bit i-1 set in
# row_bits}, capped at `cap` (early exit once reached). For {-1,0,1} matrices
# the support pattern is M mod 2, and GF(2) rank never exceeds rational rank
# (a k×k submatrix nonsingular mod 2 has odd — hence nonzero — determinant),
# so this is a sound, word-parallel lower bound used to prune rank checks.
@inline function _gf2_rank_capped(srow::Vector{UInt16}, row_bits::UInt16,
                                   colmask::UInt16, cap::Int)::Int
    cap <= 0 && return 0
    rank = 0
    # XOR basis, kept in decreasing value order; with unique leading bits,
    # value order equals leading-bit order, so one reduction pass suffices.
    b1 = UInt16(0); b2 = UInt16(0); b3 = UInt16(0)
    bits = row_bits
    @inbounds while !iszero(bits)
        i = trailing_zeros(bits) + 1
        bits &= bits - UInt16(1)
        w = srow[i] & colmask
        (b1 != 0 && xor(w, b1) < w) && (w ⊻= b1)
        (b2 != 0 && xor(w, b2) < w) && (w ⊻= b2)
        (b3 != 0 && xor(w, b3) < w) && (w ⊻= b3)
        iszero(w) && continue
        rank += 1
        rank >= cap && return rank
        if w > b1
            b3 = b2; b2 = b1; b1 = w
        elseif w > b2
            b3 = b2; b2 = w
        else
            b3 = w
        end
    end
    rank
end

# Fill buf[1..nr, 1..nc] with M[rows[1..nr], cols[1..nc]] and return rank.
# No heap allocation — buf is a caller-owned scratch buffer.
@inline function _rank_submat!(buf::Matrix{Float64}, M::Matrix{Int},
                                rows::Vector{Int}, nr::Int,
                                cols::Vector{Int}, nc::Int)::Int
    (nr == 0 || nc == 0) && return 0
    @inbounds for ci in 1:nc, ri in 1:nr
        buf[ri, ci] = M[rows[ri], cols[ci]]
    end
    _rank_float_view!(buf, nr, nc)
end

# Cached wrappers: same result as _rank_IM(M, m, mask) but avoids recomputing
# when the same column set appears multiple times in the O(N^8) decompose loop.
# Int8 array: rank ≤ N ≤ 20, so rank+1 ≤ 21 fits in Int8. Using Int8 instead of
# Int shrinks the hot working set from ~110KB to ~14KB, keeping it in L1 cache.
@inline function _rank_IM_cached(cache::Vector{Int8}, M::Matrix{Int}, m::Int, mask::UInt64)::Int
    idx = Int(mask) + 1
    v = cache[idx]
    if v == 0  # 0 = not yet computed; ranks stored as rank+1
        v = Int8(_rank_IM(M, m, mask) + 1)
        cache[idx] = v
    end
    Int(v) - 1
end
@inline function _rank_IM_cached(cache::Dict{UInt64,Int}, M::Matrix{Int}, m::Int, mask::UInt64)::Int
    v = get(cache, mask, -1)
    if v == -1
        v = _rank_IM(M, m, mask)
        cache[mask] = v
    end
    v
end

# Bitmask version — avoids vector argument allocation entirely.
# mask bit k-1 set means column k of [I_m | M] is included.
# Uses Float64 Gaussian elimination (exact for {-1,0,1} matrices, 3-5x faster
# than Bareiss due to FP multiply-subtract replacing integer division).
function _rank_IM(M::Matrix{Int}, m::Int, mask::UInt64)::Int
    iszero(mask) && return 0
    full_I = m < 64 ? (UInt64(1) << m) - UInt64(1) : typemax(UInt64)
    I_mask = mask & full_I           # bits for included I-cols
    M_bits = mask >> m               # bits for included M-cols
    n_I    = count_ones(I_mask)
    n_M    = count_ones(M_bits)
    n_M == 0 && return n_I
    n_notI = m - n_I
    n_notI == 0 && return m
    B = Matrix{Float64}(undef, n_notI, n_M)
    not_I  = full_I & ~I_mask        # bits for non-included I-rows → row indices of B
    row = 0
    nib = not_I
    @inbounds while !iszero(nib)
        ibit = nib & -nib; nib &= nib - 1
        i = trailing_zeros(ibit) + 1
        row += 1
        col = 0
        mb = M_bits
        while !iszero(mb)
            mbit = mb & -mb; mb &= mb - 1
            col += 1
            B[row, col] = M[i, trailing_zeros(mbit) + 1]
        end
    end
    n_I + _rank_float!(B)
end

"""
    _reduce(M)

Reduce matrix `M` by repeatedly:
1. Checking all entries are in {-1, 0, 1} — returns `(false, M)` if not
2. Dropping trivial rows and columns (zero or standard basis vectors)
3. Dropping linearly dependent rows and columns (equal or opposite pairs)

Returns `(true, reduced_matrix)` if successful, `(false, M)` if entries
are outside {-1, 0, 1}.
"""
function _reduce(M::Matrix{Int})::Tuple{Bool, Matrix{Int}}
    all(m -> m in (-1, 0, 1), M) || return (false, M)
    while true
        N = _drop_trivial_vectors(_drop_trivial_vectors(M, 1), 2)
        N = _drop_dependent_vectors(N)
        N == M && return (true, M)
        M = N
    end
end

# Find connected components of the support bipartite graph of M
# (vertices = rows ∪ columns, edges = nonzero entries).
# Returns nothing when the graph is connected (fast path for 2-connected matrices).
# Otherwise returns a vector of (row_indices, col_indices) pairs — one per component.
# This detects 1-sum structure in O(m·n) without any expensive rank computation.
function _bipartite_components(M::Matrix{Int})
    m, n = size(M)
    row_comp = zeros(Int, m)
    col_comp = zeros(Int, n)
    n_comps  = 0
    queue    = Int[]         # positive = row vertex, negative = –(col index)

    for r0 in 1:m
        row_comp[r0] != 0 && continue
        n_comps += 1
        k = n_comps
        row_comp[r0] = k
        push!(queue, r0)
        qi = 1
        while qi <= length(queue)
            v = queue[qi]; qi += 1
            if v > 0                          # row vertex
                for j in 1:n
                    M[v, j] != 0 || continue
                    col_comp[j] != 0 && continue
                    col_comp[j] = k
                    push!(queue, -j)
                end
            else                              # column vertex (stored as –j)
                j = -v
                for i in 1:m
                    M[i, j] != 0 || continue
                    row_comp[i] != 0 && continue
                    row_comp[i] = k
                    push!(queue, i)
                end
            end
        end
        empty!(queue)
    end

    n_comps == 1 && return nothing            # already 2-connected

    comp_rows = [Int[] for _ in 1:n_comps]
    comp_cols = [Int[] for _ in 1:n_comps]
    for i in 1:m; push!(comp_rows[row_comp[i]], i); end
    for j in 1:n; col_comp[j] > 0 && push!(comp_cols[col_comp[j]], j); end
    [(comp_rows[k], comp_cols[k]) for k in 1:n_comps]
end

# Return true if all columns of M have at most 2 nonzeros.
_all_columns_few_nonzeros(M::Matrix{Int}) =
    all(j -> count(!iszero, M[:, j]) <= 2, 1:size(M, 2))

# Build the undirected graph G on rows for Case 1.
# Vertices are rows 1..m.
# For each column with exactly 2 nonzeros in rows i and j:
#   - same sign: add edge (i,j)
#   - opposite sign: add path of length 2 via new intermediate vertex
# Returns a Graphs.SimpleGraph.
function _build_row_graph(M::Matrix{Int})
    m, n = size(M)
    # We may need up to m + n extra vertices for intermediate nodes
    g = Graphs.SimpleGraph(m + n)
    next_vertex = m + 1  # first intermediate vertex index

    for j in 1:n
        rows = findall(!iszero, M[:, j])
        length(rows) == 2 || continue
        i, k = rows[1], rows[2]
        if M[i, j] == M[k, j]  # same sign
            Graphs.add_edge!(g, i, k)
        else  # opposite sign — path of length 2
            Graphs.add_edge!(g, i, next_vertex)
            Graphs.add_edge!(g, next_vertex, k)
            next_vertex += 1
        end
    end
    return g
end

# Case 1: test if M is a network matrix when all columns have ≤2 nonzeros.
# M is a network matrix iff the row graph G is bipartite.
function _is_network_matrix_few_nonzeros(M::Matrix{Int})
    g = _build_row_graph(M)
    return Graphs.is_bipartite(g)
end

# Build graph G_i for row index i.
# Vertices are 1..m with i removed — we map them to 1..m-1.
# Returns (graph, vertex_map) where vertex_map[v] gives the original row index.
function _build_gi(M::Matrix{Int}, i::Int)
    m, n = size(M)
    # Map original row indices to graph vertices
    orig = [j for j in 1:m if j != i]  # orig[v] = original row index
    idx = zeros(Int, m)
    for (v, j) in enumerate(orig)
        idx[j] = v  # idx[j] = vertex for row j
    end

    g = Graphs.SimpleGraph(m - 1)
    for col in 1:n
        M[i, col] == 0 || continue  # skip columns with nonzero in row i
        rows = findall(!iszero, M[:, col])
        # Add edges between all pairs of rows with nonzeros in this column
        for a in 1:length(rows), b in a+1:length(rows)
            Graphs.add_edge!(g, idx[rows[a]], idx[rows[b]])
        end
    end
    return g, orig
end

# Simple BFS connected-components for a Graphs.SimpleGraph.
# Avoids Graphs.jl's connected_components overhead (vect allocations) on small graphs.
function _bfs_components(g::Graphs.SimpleGraph)
    n = Graphs.nv(g)
    visited = zeros(Bool, n)
    comps = Vector{Int}[]
    for start in 1:n
        visited[start] && continue
        comp = Int[start]
        visited[start] = true
        qi = 1
        while qi <= length(comp)
            v = comp[qi]; qi += 1
            for w in Graphs.neighbors(g, v)
                visited[w] && continue
                visited[w] = true
                push!(comp, w)
            end
        end
        push!(comps, comp)
    end
    comps
end

# Find the first row index i for which G_i is disconnected.
# Returns (i, graph, components, vertex_map) or nothing if all G_i are connected.
function _find_disconnected_gi(M::Matrix{Int})
    m = size(M, 1)
    for i in 1:m
        g, orig = _build_gi(M, i)
        comps = _bfs_components(g)
        length(comps) > 1 && return (i, g, comps, orig)
    end
    return nothing
end

"""
    _compute_w_sets(M, i, components, orig)

Compute the sets W, W_rows and U used in the network matrix recognition
algorithm (Case 2), given that G_i is disconnected.

- W = column indices where row `i` of `M` is nonzero
- W_rows[j] = W ∩ support of row `j` (for j ≠ i)
- U[k] = ∪{W_rows[j] | j ∈ components[k]}

# Arguments
- `M`: The matrix being tested
- `i`: The pivot row index (the row for which G_i is disconnected)
- `components`: Connected components of G_i as vectors of vertex indices
- `orig`: Mapping from vertex index to original row index in M

# Returns
`(W, W_rows, U)` where W and each U[k] are `Set{Int}` and W_rows is a
`Dict{Int, Set{Int}}`.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function _compute_w_sets(M::Matrix{Int}, i::Int,
                          components::Vector{Vector{Int}},
                          orig::Vector{Int})
    m, n = size(M)

    # W = support of row i
    W = Set(findall(!iszero, M[i, :]))

    # W_j = W ∩ support of row j, for each j ≠ i
    W_rows = Dict{Int, Set{Int}}()
    for j in 1:m
        j == i && continue
        W_rows[j] = W ∩ Set(findall(!iszero, M[j, :]))
    end

    # U_k = union of W_j for all j in component k
    U = Vector{Set{Int}}(undef, length(components))
    for (k, component) in enumerate(components)
        U_k = Set{Int}()
        for v in component
            union!(U_k, W_rows[orig[v]])
        end
        U[k] = U_k
    end

    return W, W_rows, U
end

"""
    _build_h(components, orig, W_rows, U)

Build the graph H on components C_1,...,C_p of G_i.

Two components C_k and C_l are adjacent in H iff:
- ∃ i ∈ C_k : U_k ⊄ W_i and U_k ∩ W_i ≠ ∅, and
- ∃ j ∈ C_l : U_l ⊄ W_j and U_l ∩ W_j ≠ ∅

# Arguments
- `components`: Connected components of G_i
- `orig`: Mapping from vertex index to original row index
- `W_rows`: Dict mapping row index => W ∩ support(row)
- `U`: Vector of sets, U[k] = ∪{W_rows[j] | j ∈ components[k]}

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function _build_h(components::Vector{Vector{Int}},
                  orig::Vector{Int},
                  W_rows::Dict{Int, Set{Int}},
                  U::Vector{Set{Int}})
    p = length(components)
    h = Graphs.SimpleGraph(p)

    for k in 1:p, l in k+1:p
        # Check: ∃ i ∈ C_k such that U_l ⊄ W_i and U_l ∩ W_i ≠ ∅
        k_ok = any(components[k]) do v
            j = orig[v]
            !issubset(U[l], W_rows[j]) && !isempty(U[l] ∩ W_rows[j])
        end
        k_ok || continue

        # Check: ∃ j ∈ C_l such that U_k ⊄ W_j and U_k ∩ W_j ≠ ∅
        l_ok = any(components[l]) do v
            j = orig[v]
            !issubset(U[k], W_rows[j]) && !isempty(U[k] ∩ W_rows[j])
        end
        l_ok || continue

        Graphs.add_edge!(h, k, l)
    end
    return h
end

"""
    _split_submatrices(M, i, components, orig)

Extract submatrices M_1,...,M_p from `M`, where each M_k consists of:
- Row `i` (the pivot row, i.e. the row for which G_i is disconnected)
- All rows of `M` with index in component `k`

# Arguments
- `M`: The matrix being tested
- `i`: The pivot row index
- `components`: Connected components of G_i as vectors of vertex indices
- `orig`: Mapping from vertex index to original row index in M

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function _split_submatrices(M::Matrix{Int}, i::Int,
                             components::Vector{Vector{Int}},
                             orig::Vector{Int})
    return [M[[i; [orig[v] for v in component]], :] for component in components]
end

"""
    _is_network_matrix(M)

Test whether integer matrix `M` is a network matrix using the recursive
algorithm of Theorem 20.1.

A matrix is a network matrix if it can be represented by a directed tree `T`
and digraph `D`, where entry M[a', a] encodes how the unique path in `T`
between the endpoints of arc `a ∈ D` traverses arc `a' ∈ T`: +1 forwardly,
-1 backwardly, 0 not at all.

The algorithm proceeds in two cases:

**Case 1:** If all columns of `M` have at most two nonzeros, `M` is a network
matrix if and only if the row graph G is bipartite.

**Case 2:** If some column has three or more nonzeros, find a row index `i`
for which G_i is disconnected. If no such `i` exists, `M` is not a network
matrix. Otherwise, build the graph H on the connected components of G_i —
`M` is a network matrix if and only if H is bipartite and each submatrix
M_k is a network matrix (recursively).

# Arguments
- `M::Matrix{Int}`: An integer matrix with entries in {-1, 0, 1}.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.1.
"""
function _is_network_matrix(M::Matrix{Int})
    # Case 1: all columns have ≤2 nonzeros
    if _all_columns_few_nonzeros(M)
        return _is_network_matrix_few_nonzeros(M)
    end

    # Case 2: some column has ≥3 nonzeros
    result = _find_disconnected_gi(M)

    # All G_i connected → not a network matrix
    result === nothing && return false

    i, g, components, orig = result
    W, W_rows, U = _compute_w_sets(M, i, components, orig)
    h = _build_h(components, orig, W_rows, U)

    # H must be bipartite
    Graphs.is_bipartite(h) || return false

    # Recursively test each submatrix
    submatrices = _split_submatrices(M, i, components, orig)
    return all(_is_network_matrix, submatrices)
end

# ──────────────────────────────────────────────────────────────────────────────
# Pivot operation
# ──────────────────────────────────────────────────────────────────────────────

"""
    pivot(M, k)

Perform the pivot operation on matrix `M` with respect to its leading k×k
submatrix E, which must be invertible with determinant ±1 (as holds for
submatrices of TU matrices).

Given the partition M = [E C; B D], returns:

    [-E⁻¹    E⁻¹C  ]
    [ BE⁻¹   D-BE⁻¹C]

This operation preserves total unimodularity and is central to Seymour
decomposition.

# Arguments
- `M::Matrix{Int}`: An integer matrix whose entries are in {-1, 0, 1}.
- `k::Int`: Size of the leading square submatrix to pivot on.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function pivot(M::Matrix{Int}, k::Int)
    @views E = M[1:k,     1:k    ]
    @views B = M[k+1:end, 1:k    ]
    @views C = M[1:k,     k+1:end]
    @views D = M[k+1:end, k+1:end]
    Einv = Matrix{Int}(round.(inv(Matrix{Rational{Int}}(E))))
    [-Einv        Einv*C
      B*Einv  D - B*Einv*C]
end

"""
    _decompose(M)

Test whether the rows and columns of `M` can be permuted so that

    M = [A  B]
        [C  D]

with rank(B) + rank(C) ≤ 2 and both A and D having r + c ≥ 4.

For matrices with m ≤ 12 and n ≤ 12: enumerate all 2^m × 2^n row/column
bipartitions directly.  This is O(2^m × 2^n × rank) but with aggressive
early exit (most splits fail on rank(B) alone) and is both simpler and more
correct than the matroid-intersection approach.

For larger matrices: falls back to the matroid-intersection algorithm
(Theorem 20.2), which may miss some decompositions but is conservative
(only gives false negatives, never false positives).

Returns `(true, (A, B, C, D))` if such a decomposition exists,
or `(false, (M, M, M, M))` if not.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.2.
"""
function _decompose(M::Matrix{Int};
                    reject_degenerate_3sum::Bool = false)::Tuple{Bool, NTuple{4, Matrix{Int}}}
    m, n = size(M)

    if m <= 12 && n <= 12
        # Allocation-free bipartition search.
        # All index buffers and the Float64 scratch buffer are hoisted outside every
        # loop level — zero heap allocation inside the O(4^m) hot path.
        row_top   = Vector{Int}(undef, m)
        row_bot   = Vector{Int}(undef, m)
        col_left  = Vector{Int}(undef, n)
        col_right = Vector{Int}(undef, n)
        buf       = Matrix{Float64}(undef, m, n)   # scratch for rank computation

        # Row support patterns as bitmasks (bit j-1 ⇔ M[i,j] ≠ 0), for the
        # GF(2) rank prefilter: most candidate bipartitions die on a
        # word-parallel GF(2) lower bound without ever touching the Float64
        # rank path or building column index lists.
        srow = Vector{UInt16}(undef, m)
        for i in 1:m
            s = UInt16(0)
            for j in 1:n; iszero(M[i, j]) || (s |= UInt16(1) << (j - 1)); end
            srow[i] = s
        end
        full_rows = (UInt16(1) << m) - UInt16(1)
        full_cols = (UInt16(1) << n) - UInt16(1)

        # Two-pass search: pass 1 only accepts rB+rC ≤ 1 (2-sums), pass 2
        # accepts rB+rC ≤ 2 (3-sums and pivots). Preferring 2-sums avoids
        # choosing pivots that can cause the recursion to cycle back to a
        # matrix already in `seen`.
        for pass in 1:2
            max_sum = pass == 1 ? 1 : 2
            for rt_mask in UInt16(1):(UInt16(1) << m) - UInt16(2)
                nrt = count_ones(rt_mask); nrb = m - nrt
                rb_mask = full_rows & ~rt_mask
                nt = 0; nb = 0
                rows_built = false
                for cl_mask in UInt16(1):(UInt16(1) << n) - UInt16(2)
                    ncl = count_ones(cl_mask); ncr = n - ncl
                    nrt + ncl >= 4 || continue
                    nrb + ncr >= 4 || continue
                    cr_mask = full_cols & ~cl_mask
                    # GF(2) lower bounds: rank_GF2 ≤ rank_ℚ, so exceeding the
                    # budget here rules the candidate out for certain.
                    gB = _gf2_rank_capped(srow, rt_mask, cr_mask, max_sum + 1)
                    gB > max_sum && continue
                    gC = _gf2_rank_capped(srow, rb_mask, cl_mask, max_sum - gB + 1)
                    gB + gC > max_sum && continue
                    if !rows_built
                        for i in 1:m
                            if (rt_mask >> (i-1)) & 1 == 1; row_top[nt += 1] = i
                            else;                            row_bot[nb += 1] = i; end
                        end
                        rows_built = true
                    end
                    nl = 0; nr = 0
                    for j in 1:n
                        if (cl_mask >> (j-1)) & 1 == 1; col_left[nl += 1] = j
                        else;                            col_right[nr += 1] = j; end
                    end
                    rB = _rank_submat!(buf, M, row_top, nrt, col_right, ncr)
                    rB > max_sum && continue
                    rC = _rank_submat!(buf, M, row_bot, nrb, col_left, ncl)
                    rB + rC > max_sum && continue
                    # When requested, skip 3-sum partitions whose A or D subblock is
                    # degenerate — _apply_decomposition needs non-degenerate A/D to
                    # construct the Case 4 matrices correctly.
                    if reject_degenerate_3sum && rB == 1 && rC == 1
                        (_is_degenerate(M[row_top[1:nrt], col_left[1:ncl]]) ||
                         _is_degenerate(M[row_bot[1:nrb], col_right[1:ncr]])) && continue
                    end
                    return (true, (M[row_top[1:nrt], col_left[1:ncl]],
                                   M[row_top[1:nrt], col_right[1:ncr]],
                                   M[row_bot[1:nrb], col_left[1:ncl]],
                                   M[row_bot[1:nrb], col_right[1:ncr]]))
                end
            end
        end
        return (false, (M, M, M, M))
    end

    # Fall back: matroid-intersection approach for larger matrices.
    return _decompose_matroid(M; reject_degenerate_3sum)
end

# Matroid-intersection fallback for matrices that exceed the bipartition threshold.
function _decompose_matroid(M::Matrix{Int};
                             reject_degenerate_3sum::Bool = false)::Tuple{Bool, NTuple{4, Matrix{Int}}}
    m, n = size(M)
    N = m + n
    rhoX = m

    valid_masks = UInt64[]
    for s in combinations(1:N, 4)
        has_I = false; has_M = false
        for x in s
            x <= m ? (has_I = true) : (has_M = true)
            has_I & has_M && break
        end
        has_I & has_M || continue
        mask = UInt64(0)
        for x in s; mask |= UInt64(1) << (x - 1); end
        push!(valid_masks, mask)
    end

    slow_dbuf = Vector{Tuple{Int,Int}}(undef, 2 * N * N)
    slow_prev = Vector{Int}(undef, N)
    slow_bfsq = Vector{Int}(undef, N)
    if N <= 20
        return _decompose_loop(M, m, n, rhoX, valid_masks, zeros(Int8, 1 << N),
                               slow_dbuf, slow_prev, slow_bfsq;
                               reject_degenerate_3sum)
    else
        return _decompose_loop(M, m, n, rhoX, valid_masks, Dict{UInt64, Int}(),
                               slow_dbuf, slow_prev, slow_bfsq;
                               reject_degenerate_3sum)
    end
end

function _decompose_loop(M::Matrix{Int}, m::Int, n::Int, rhoX::Int,
                          valid_masks::Vector{UInt64}, cache,
                          slow_dbuf::Vector{Tuple{Int,Int}},
                          slow_prev::Vector{Int},
                          slow_bfsq::Vector{Int};
                          reject_degenerate_3sum::Bool = false)::Tuple{Bool, NTuple{4, Matrix{Int}}}
    @inbounds for S_mask in valid_masks
        @inbounds for T_mask in valid_masks
            # S and T must be disjoint
            S_mask & T_mask != 0 && continue

            # Solve problem (16) — returns Y as a bitmask
            found, Y_mask = _solve_submodular(M, m, S_mask, T_mask, rhoX, cache,
                                              slow_dbuf, slow_prev, slow_bfsq)
            found || continue

            # Y∩I-cols → row indices for top partition; Y∩M-cols → col indices for left partition
            row_top   = [i for i in 1:m if (Y_mask >> (i-1))   & 1 == 1]
            col_left  = [j for j in 1:n if (Y_mask >> (m+j-1)) & 1 == 1]
            row_bot   = [i for i in 1:m if (Y_mask >> (i-1))   & 1 == 0]
            col_right = [j for j in 1:n if (Y_mask >> (m+j-1)) & 1 == 0]

            # Size constraints: A and D must have r + c ≥ 4
            length(row_top)  + length(col_left)  >= 4 || continue
            length(row_bot)  + length(col_right) >= 4 || continue

            # Must have at least one row and column in each partition
            isempty(row_top)  && continue
            isempty(row_bot)  && continue
            isempty(col_left) && continue
            isempty(col_right) && continue

            # Extract submatrices.
            # Note: rank(B)+rank(C) = rhoY+rhoXminusY-m ≤ 2 is already guaranteed
            # by found=true from _solve_submodular, so no extra rank check needed.
            A = M[row_top,  col_left ]
            B = M[row_top,  col_right]
            C = M[row_bot,  col_left ]
            D = M[row_bot,  col_right]

            if reject_degenerate_3sum
                rB_chk = _rank_int(B); rC_chk = _rank_int(C)
                if rB_chk == 1 && rC_chk == 1
                    (_is_degenerate(A) || _is_degenerate(D)) && continue
                end
            end

            return (true, (A, B, C, D))
        end
    end
    return (false, (M, M, M, M))
end

# All column sets are represented as UInt64 bitmasks throughout, eliminating
# the vector-union allocations (S∪Z, SZ∪[v], S∪Zminu∪[v], etc.) that previously
# dominated the allocation count in the O(N^8) outer loop.
# rank_cache is shared across all (S,T) pairs in _decompose — the same column-
# subset rank is queried many times, so caching eliminates redundant Bareiss runs.
function _solve_submodular(M::Matrix{Int}, m::Int, S_mask::UInt64,
                            T_mask::UInt64, rhoX::Int, cache,
                            d_buf::Vector{Tuple{Int,Int}},
                            prev::Vector{Int},
                            bfsq::Vector{Int})::Tuple{Bool, UInt64}
    rk(mask) = _rank_IM_cached(cache, M, m, mask)

    N = m + size(M, 2)
    all_mask = N < 64 ? (UInt64(1) << N) - UInt64(1) : typemax(UInt64)
    ST_mask = S_mask | T_mask
    Z_mask = UInt64(0)

    # First iteration fast path: Z=∅ means no edges exist.
    # Iterate only over active bits (V\(S∪T)) to avoid skip-checks for all N.
    let
        rhoSZ = rk(S_mask)
        rhoTZ = rk(T_mask)
        U_mask = UInt64(0); W_mask = UInt64(0)
        vbits = all_mask & ~ST_mask  # V \ (S∪T)
        vb = vbits
        while !iszero(vb)
            vbit = vb & -vb; vb &= vb - 1  # pop lowest set bit
            rk(S_mask | vbit) == rhoSZ + 1 && (U_mask |= vbit)
            rk(T_mask | vbit) == rhoTZ + 1 && (W_mask |= vbit)
        end
        UW = U_mask & W_mask
        if !iszero(UW)
            Z_mask = UW & -UW  # lowest set bit of U∩W → length-1 augmenting path
        else
            XminusY_mask = all_mask & ~S_mask
            rhoXminusY = iszero(XminusY_mask) ? 0 : rk(XminusY_mask)
            return (rhoSZ + rhoXminusY <= rhoX + 2, S_mask)
        end
    end

    while true
        SZ_mask = S_mask | Z_mask
        TZ_mask = T_mask | Z_mask
        rhoSZ = rk(SZ_mask)
        rhoTZ = rk(TZ_mask)

        # Iterate only over V\(S∪T∪Z) — avoids N skip-checks per active element
        U_mask = UInt64(0); W_mask = UInt64(0)
        let vb = all_mask & ~ST_mask & ~Z_mask
            while !iszero(vb)
                vbit = vb & -vb; vb &= vb - 1
                rk(SZ_mask | vbit) == rhoSZ + 1 && (U_mask |= vbit)
                rk(TZ_mask | vbit) == rhoTZ + 1 && (W_mask |= vbit)
            end
        end

        # Build digraph D: iterate over Z bits and V\(S∪T∪Z) bits directly.
        # Edge condition: swapping u∈Z for v∈V\(S∪T∪Z) keeps rank(S∪Z) unchanged
        # → rk(S∪(Z\{u})∪{v}) = rk(S∪Z) = rhoSZ (and analogously for T).
        # Using rhoSZ/rhoTZ (not rhoS+|Z|) is correct even when the rank
        # invariant rk(S∪Z)=rk(S)+|Z| fails after non-trivial augmenting paths.
        n_de = 0
        VnotSTZ = all_mask & ~ST_mask & ~Z_mask
        let zb = Z_mask
            while !iszero(zb)
                u_lsb = zb & -zb; zb &= zb - 1  # pop lowest Z-bit
                u = trailing_zeros(u_lsb) + 1
                Zminu_mask = Z_mask & ~u_lsb
                SZminu = S_mask | Zminu_mask
                TZminu = T_mask | Zminu_mask
                rhoSZ_target = rhoSZ
                rhoTZ_target = rhoTZ
                let vb = VnotSTZ
                    while !iszero(vb)
                        vbit = vb & -vb; vb &= vb - 1
                        if rk(SZminu | vbit) == rhoSZ_target
                            n_de += 1; d_buf[n_de] = (u, trailing_zeros(vbit) + 1)
                        end
                        if rk(TZminu | vbit) == rhoTZ_target
                            n_de += 1; d_buf[n_de] = (trailing_zeros(vbit) + 1, u)
                        end
                    end
                end
            end
        end

        found, path_mask = _shortest_path_mask(d_buf, n_de, U_mask, W_mask, N, prev, bfsq)

        if found
            Z_mask = Z_mask ⊻ path_mask
        else
            reach_mask = _reachable_bitmask(d_buf, n_de, W_mask, ST_mask, N, bfsq)
            Y_mask = S_mask | reach_mask

            XminusY_mask = all_mask & ~Y_mask
            rhoY       = iszero(reach_mask) ? rhoSZ : rk(Y_mask)
            rhoXminusY = iszero(XminusY_mask) ? 0 : rk(XminusY_mask)

            return (rhoY + rhoXminusY <= rhoX + 2, Y_mask)
        end
    end
end

# BFS augmenting path using bitmasks + pre-allocated prev/queue arrays.
# Returns (found, path_mask) where path_mask is the symmetric-difference to
# apply to Z (all vertices on the augmenting path, including endpoints).
function _shortest_path_mask(edges::Vector{Tuple{Int,Int}}, n_edges::Int,
                               U_mask::UInt64, W_mask::UInt64, N::Int,
                               prev::Vector{Int}, queue::Vector{Int})::Tuple{Bool, UInt64}
    iszero(U_mask) && return (false, UInt64(0))
    iszero(W_mask) && return (false, UInt64(0))

    # Length-0 augmenting path: element in both U and W
    UW = U_mask & W_mask
    if !iszero(UW)
        v = trailing_zeros(UW) + 1
        return (true, UInt64(1) << (v - 1))
    end

    # BFS: sources = U_mask, targets = W_mask
    visited = U_mask
    @inbounds for i in 1:N; prev[i] = 0; end
    qhead = 1; qtail = 0
    let ub = U_mask
        while !iszero(ub)
            bit = ub & -ub; ub &= ub - 1
            qtail += 1; queue[qtail] = trailing_zeros(bit) + 1
        end
    end

    while qhead <= qtail
        v = queue[qhead]; qhead += 1
        for ei in 1:n_edges
            u, w = edges[ei]
            u == v || continue
            (visited >> (w - 1)) & 1 == 1 && continue
            visited |= UInt64(1) << (w - 1)
            prev[w] = v
            if (W_mask >> (w - 1)) & 1 == 1
                # Reconstruct path_mask (no allocation — just bitmask accumulation)
                path_mask = UInt64(1) << (w - 1)
                curr = w
                while (U_mask >> (curr - 1)) & 1 == 0
                    curr = prev[curr]
                    path_mask |= UInt64(1) << (curr - 1)
                end
                return (true, path_mask)
            end
            qtail += 1; queue[qtail] = w
        end
    end
    return (false, UInt64(0))
end

# BFS on reversed graph from W_mask; returns bitmask of reachable non-W elements.
# queue buffer is passed in (reuse the bfsq from the caller — safe because
# _shortest_path_mask has already returned false, so bfsq is unused).
function _reachable_bitmask(edges::Vector{Tuple{Int,Int}}, n_edges::Int,
                              W_mask::UInt64, ST_mask::UInt64, N::Int,
                              queue::Vector{Int})::UInt64
    iszero(W_mask) && return UInt64(0)
    visited = W_mask
    qhead = 1; qtail = 0
    let wb = W_mask
        while !iszero(wb)
            bit = wb & -wb; wb &= wb - 1
            qtail += 1; queue[qtail] = trailing_zeros(bit) + 1
        end
    end
    while qhead <= qtail
        v = queue[qhead]; qhead += 1
        for ei in 1:n_edges
            u, w = edges[ei]
            w == v || continue
            (visited >> (u - 1)) & 1 == 1 && continue
            visited |= UInt64(1) << (u - 1)
            qtail += 1; queue[qtail] = u
        end
    end
    # Return visited \ W_mask, restricted to V \ ST_mask — a single bitmask expression.
    visited & ~W_mask & ~ST_mask
end

"""
    _extract_rank1(B)

Extract f and g from a rank-1 matrix B = f⊗g, where f and g are {0,±1}
vectors (f a column, g a row). f is the first nonzero column of B; since
rank(B) = 1 and all entries are in {-1,0,1}, every nonzero column of B is
either f or -f, so g[j] ∈ {+1,-1,0} accordingly and f⊗g == B exactly.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3, Case 2.
"""
function _extract_rank1(B::Matrix{Int})
    m, n = size(B)
    col = findfirst(j -> any(!iszero, B[:, j]), 1:n)
    f = B[:, col]  # no normalisation
    negf = -f
    g = zeros(Int, 1, n)
    for j in 1:n
        if @views B[:, j] == f
            g[1, j] = 1
        elseif @views B[:, j] == negf
            g[1, j] = -1
        end
    end
    return reshape(f, m, 1), g
end

"""
    _find_epsilon(A, R_rows, K_cols)

Find ε ∈ {+1,-1} for Case 4 of Theorem 20.3.

Build a bipartite graph G on rows and columns of `A`. `R_rows` and `K_cols`
are the sets of rows and columns intersecting A4. Find a shortest path Π
from R to K in G, compute δ = sum of A entries on edges of Π (which has odd
length, so δ is odd), and return:

    ε = +1 if δ ≡  1 (mod 4)
    ε = -1 if δ ≡ -1 (mod 4)

If A4 = A[R_rows, K_cols] has a nonzero entry, ε equals that entry directly.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3, Case 4.
"""
function _find_epsilon(A::Matrix{Int}, R_rows::Vector{Int}, K_cols::Vector{Int})::Tuple{Bool, Int}
    m, n = size(A)

    A4 = A[R_rows, K_cols]
    nz = findfirst(!iszero, A4)
    if nz !== nothing
        return (true, A4[nz])
    end

    sources = R_rows
    targets = m .+ K_cols
    target_set = Set(targets)

    visited = Dict{Int, Union{Nothing, Tuple{Int,Int}}}()
    for s in sources
        visited[s] = nothing
    end
    queue = copy(sources)
    found_target = nothing

    while !isempty(queue) && found_target === nothing
        v = popfirst!(queue)
        for i in 1:m, j in 1:n
            A[i, j] == 0 && continue
            r_v, c_v = i, m + j
            next = v == r_v ? c_v : (v == c_v ? r_v : nothing)
            next === nothing && continue
            next in keys(visited) && continue
            visited[next] = (v, A[i, j])
            if next in target_set
                found_target = next
                break
            end
            push!(queue, next)
        end
    end

    found_target === nothing && return (false, 0)

    delta = 0
    v = found_target
    while visited[v] !== nothing
        parent, w = visited[v]
        delta += w
        v = parent
    end

    mod4 = mod(delta, 4)
    mod4 == 1 && return (true, 1)
    mod4 == 3 && return (true, -1)
    error("δ = $delta is even — path should have odd length")
end


# ──────────────────────────────────────────────────────────────────────────────
# Seymour decomposition operations
# ──────────────────────────────────────────────────────────────────────────────

"""
    one_sum(A, B)

Compute the 1-sum of integer matrices `A` and `B`.

The 1-sum is the block diagonal matrix:

    [A  0]
    [0  B]

If `A` and `B` are both totally unimodular, so is their 1-sum.

# Arguments
- `A`, `B`: Integer matrices, each with r + c ≥ 4.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function one_sum(A::Matrix{Int}, B::Matrix{Int})
    rA, cA = _check_size(A)
    rB, cB = _check_size(B)
    C = zeros(Int, rA + rB, cA + cB)
    C[1:rA,       1:cA      ] = A
    C[rA+1:rA+rB, cA+1:cA+cB] = B
    return C
end

"""
    two_sum(A, B)

Compute the 2-sum of integer matrices `A` and `B`.

`A` must have a distinguished last column `a`, and `B` a distinguished first
row `bᵀ`. The 2-sum is:

    [Am   a⊗b]
    [0    Bm ]

where Am is A with its last column removed, and Bm is B with its first row
removed.

If `A` and `B` are both totally unimodular, so is their 2-sum.

# Arguments
- `A`, `B`: Integer matrices, each with r + c ≥ 4.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function two_sum(A::Matrix{Int}, B::Matrix{Int})
    _check_size(A)
    _check_size(B)
    @views Am, a = A[:, 1:end-1], A[:, end]
    @views b,  Bm = B[1, :],      B[2:end, :]
    rA, cA = size(Am)
    rB, cB = size(Bm)
    C = zeros(Int, rA + rB, cA + cB)
    @views C[1:rA,       1:cA      ] = Am
    @views C[rA+1:rA+rB, cA+1:cA+cB] = Bm
    @views C[1:rA,       cA+1:cA+cB] = a * b'
    return C
end

"""
    three_sum(A, B)

Compute the 3-sum of integer matrices `A` and `B`.

`A` must have the form:

    [Am   a  a]
    [cᵀ   0  1]

and `B` must have the form:

    [1  0  bᵀ]
    [d  d  Bm]

where `a`, `c`, `b`, `d` are vectors. The 3-sum combines these matrices
by eliminating the shared structure:

    [Am    a⊗bᵀ]
    [d⊗cᵀ  Bm  ]

If `A` and `B` are both totally unimodular, so is their 3-sum.

# Arguments
- `A`, `B`: Integer matrices in the required form; an error is thrown otherwise.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function three_sum(A::Matrix{Int}, B::Matrix{Int})
    _check_size(A)
    _check_size(B)
    @views begin
        (A[1:end-1, end-1] != A[1:end-1, end] ||
         A[end, end-1] != 0 || A[end, end] != 1) &&
            error("Matrix A does not have the required form for a 3-sum.")
        (B[2:end, 1] != B[2:end, 2] ||
         B[1, 1] != 1 || B[1, 2] != 0) &&
            error("Matrix B does not have the required form for a 3-sum.")
    end
    @views Am, a, c = A[1:end-1, 1:end-2], A[1:end-1, end], A[end, 1:end-2]
    @views Bm, b, d = B[2:end,   3:end  ], B[1,       3:end], B[2:end, 1]
    rA, cA = size(Am)
    rB, cB = size(Bm)
    C = zeros(Int, rA + rB, cA + cB)
    @views C[1:rA,       1:cA      ] = Am
    @views C[1:rA,       cA+1:cA+cB] = a * b'
    @views C[rA+1:rA+rB, 1:cA      ] = d * c'
    @views C[rA+1:rA+rB, cA+1:cA+cB] = Bm
    return C
end

# ──────────────────────────────────────────────────────────────────────────────
# Total unimodularity tests
# ──────────────────────────────────────────────────────────────────────────────

"""
    naive_is_totally_unimodular(M)

Test whether integer matrix `M` is totally unimodular by checking that every
square submatrix has determinant in {-1, 0, 1}.

This algorithm is correct but has exponential time complexity in the size of
`M`. It is intended for testing and validation only. See
[`is_totally_unimodular`](@ref) for the polynomial-time implementation.

# Arguments
- `M::Matrix{Int}`: An integer matrix whose entries must be in {-1, 0, 1};
  returns `false` immediately otherwise.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function naive_is_totally_unimodular(M::Matrix{Int})
    all(m -> m in (-1, 0, 1), M) || return false
    r, c = size(M)
    for s in 1:min(r, c)
        buf = Matrix{Int}(undef, s, s)
        for rows in combinations(1:r, s), cols in combinations(1:c, s)
            for (cj, j) in enumerate(cols), (ri, i) in enumerate(rows)
                buf[ri, cj] = M[i, j]
            end
            # Exact integer determinant — Float64 det could misreport ±1/0
            # near the rounding threshold, and this is the test oracle.
            _det_int!(buf) in (-1, 0, 1) || return false
        end
    end
    return true
end

"""
    is_totally_unimodular(M)

Test whether the integer matrix `M` is totally unimodular (TU), i.e. whether
every square submatrix of `M` has determinant in {-1, 0, 1}, using the
polynomial-time algorithm based on Seymour's decomposition theorem
(Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3).

Any `AbstractMatrix` with integer-valued entries is accepted; entries outside
{-1, 0, 1} make the matrix trivially non-TU, so `false` is returned.

# Example
```jldoctest
julia> is_totally_unimodular([1 0 1 0; 0 1 1 0; 0 0 1 1])
true

julia> is_totally_unimodular([1 1 0; 1 0 1; 0 1 1])   # has a 3×3 det = -2
false
```

Compare with [`naive_is_totally_unimodular`](@ref), which checks all square
submatrix determinants directly (exponential time, used as a test oracle).
"""
function is_totally_unimodular(M::Matrix{Int})::Bool
    _is_tu_recursive(M, 0, Set{Matrix{Int}}())
end

# Convenience methods: accept any integer-valued matrix (Bool, Int8, views, …).
# Entries outside {-1,0,1} mean non-TU, so reject before converting — this
# also avoids InexactError for values that don't fit in Int.
function _to_int_matrix(M::AbstractMatrix{<:Integer})::Union{Matrix{Int}, Nothing}
    all(x -> -1 <= x <= 1, M) ? Matrix{Int}(M) : nothing
end
for f in (:is_totally_unimodular, :naive_is_totally_unimodular)
    @eval function $f(M::AbstractMatrix{<:Integer})::Bool
        N = _to_int_matrix(M)
        N === nothing ? false : $f(N)
    end
end
function cmr_is_totally_unimodular(M::AbstractMatrix{<:Integer}; kwargs...)::Bool
    N = _to_int_matrix(M)
    N === nothing ? false : cmr_is_totally_unimodular(N; kwargs...)
end

function _is_tu_recursive(M::Matrix{Int}, depth::Int, seen::Set{Matrix{Int}})::Bool
    depth > 100 && error("Maximum recursion depth exceeded")

    ok, M = _reduce(M)
    ok || return false
    (size(M, 1) == 0 || size(M, 2) == 0) && return true

    # 1-sum: split into connected components and test each independently.
    # O(m·n) bipartite BFS — far cheaper than any subsequent step.
    let comps = _bipartite_components(M)
        if comps !== nothing
            return all(((rows, cols),) -> _is_tu_recursive(M[rows, cols], depth+1, seen), comps)
        end
    end

    # Cycle detection: `seen` holds the matrices on the *current* recursion
    # path only. Membership means the pivot cases (5/6) have cycled back to an
    # ancestor, so this branch cannot make progress. Each matrix is removed
    # once its subtree is done — identical matrices in sibling branches (e.g.
    # duplicate blocks of a 1-sum) are legitimate and must not be rejected.
    M in seen && return false
    push!(seen, M)
    result = _is_tu_irreducible(M, depth, seen)
    delete!(seen, M)
    return result
end

# TU test for a matrix that is already reduced, connected, and not on the
# current recursion path.
function _is_tu_irreducible(M::Matrix{Int}, depth::Int, seen::Set{Matrix{Int}})::Bool
    _is_network_matrix(M) && return true
    _is_network_matrix(Matrix{Int}(M')) && return true
    _is_special_matrix(M) && return true

    # Quick non-TU detector: Eulerian check at k ≤ 3 catches most violations
    # (e.g. any 2×2 or 3×3 bad submatrix) in microseconds, before the O(4^m)
    # bipartition search runs.
    _tu_eulerian(M, 3) || return false

    # Matrices beyond the 12×12 bipartition threshold would fall into the
    # matroid-intersection search, which enumerates ~C(m+n,4)² (S,T) pairs —
    # hours for a 14×14 matrix. When the smaller dimension is modest, the
    # exact branch-and-prune Ghouila-Houri test answers far faster: measured
    # worst cases (dense TU matrices, which force exhaustion) are ~0.5s at
    # min-dim 18, ~3s at 20, ~13s at 22; non-TU inputs usually exit in
    # milliseconds. Beyond the cap, the matroid search is the only option —
    # and impractically slow, see IMPLEMENTATION_NOTES.md.
    m, n = size(M)
    if (m > 12 || n > 12) && min(m, n) <= 24
        return _tu_partition(M)
    end

    found, (A, B, C, D) = _decompose(M)
    found || return false

    rB = _rank_int(B)
    rC = _rank_int(C)
    return _apply_decomposition(M, A, B, C, D, rB, rC, depth, seen)
end

# Dispatch on the rank case of a decomposition M = [A B; C D].
# When rB==1 and rC==1 (3-sum / Case 4), if the first decomposition found by
# _decompose gives a degenerate A or D, we retry with reject_degenerate_3sum=true
# to find a non-degenerate partition instead of incorrectly returning false.
function _apply_decomposition(M::Matrix{Int},
                               A::Matrix{Int}, B::Matrix{Int},
                               C::Matrix{Int}, D::Matrix{Int},
                               rB::Int, rC::Int,
                               depth::Int, seen::Set{Matrix{Int}})::Bool
    if rB == 0 && rC == 0
        return _is_tu_recursive(A, depth+1, seen) && _is_tu_recursive(D, depth+1, seen)

    elseif rB == 1 && rC == 0
        f, g = _extract_rank1(B)
        return _is_tu_recursive([A f], depth+1, seen) &&
               _is_tu_recursive([g; D], depth+1, seen)

    elseif rB == 0 && rC == 1
        f, g = _extract_rank1(C)
        return _is_tu_recursive([A; g], depth+1, seen) &&
               _is_tu_recursive([f D], depth+1, seen)

    elseif rB == 1 && rC == 1
        # If the current partition is degenerate (A or D has trivial/dependent
        # rows or columns), it is not suitable for the 3-sum construction.
        # Search for an alternative non-degenerate partition instead.
        if _is_degenerate(A) || _is_degenerate(D)
            found2, (A2, B2, C2, D2) = _decompose(M; reject_degenerate_3sum = true)
            if !found2
                # Every rB+rC≤2 partition has degenerate A/D. For small matrices,
                # fall back to the partition algorithm (exponential but correct).
                size(M, 1) <= 12 && size(M, 2) <= 12 && return _tu_partition(M)
                return false
            end
            rB2 = _rank_int(B2); rC2 = _rank_int(C2)
            return _apply_decomposition(M, A2, B2, C2, D2, rB2, rC2, depth, seen)
        end

        f_B, g_B = _extract_rank1(B)
        f_C, g_C = _extract_rank1(C)
        B_rows    = findall(!iszero, f_B[:, 1])
        C_cols    = findall(!iszero, g_C[1, :])
        notB_rows = [i for i in 1:size(A, 1) if i ∉ B_rows]
        notC_cols = [j for j in 1:size(A, 2) if j ∉ C_cols]
        C_rows    = findall(!iszero, f_C[:, 1])
        B_cols    = findall(!iszero, g_B[1, :])
        notC_rows = [i for i in 1:size(D, 1) if i ∉ C_rows]
        notB_cols = [j for j in 1:size(D, 2) if j ∉ B_cols]
        # Scale M so that both B and C become all-ones on their supports
        # (Schrijver's standard form (28)): rows i ∈ B_rows by f_B[i], rows
        # i ∈ C_rows by f_C[i], columns j ∈ C_cols by g_C[j], columns
        # j ∈ B_cols by g_B[j]. Row scalings hit A/D rows; column scalings
        # hit A's C_cols and D's B_cols. TU is invariant under ±1 scalings.
        A_norm = copy(A)
        for i in B_rows
            A_norm[i, :] *= f_B[i, 1]
        end
        for j in C_cols
            A_norm[:, j] *= g_C[1, j]
        end
        D_norm = copy(D)
        for i in C_rows
            D_norm[i, :] *= f_C[i, 1]
        end
        for j in B_cols
            D_norm[:, j] *= g_B[1, j]
        end
        A1 = A_norm[notB_rows, notC_cols]
        A2 = A_norm[notB_rows, C_cols   ]
        A3 = A_norm[B_rows,    notC_cols]
        A4 = A_norm[B_rows,    C_cols   ]
        D1 = D_norm[C_rows,    B_cols   ]
        D2 = D_norm[C_rows,    notB_cols]
        D3 = D_norm[notC_rows, B_cols   ]
        D4 = D_norm[notC_rows, notB_cols]
        ok1, ε₁ = _find_epsilon(A_norm, B_rows, C_cols)
        ok2, ε₂ = _find_epsilon(D_norm, C_rows, B_cols)
        (ok1 && ok2) || return false
        nR     = length(B_rows)
        nK     = length(C_cols)
        nnotR  = length(notB_rows)
        nnotK  = length(notC_cols)
        nCR    = length(C_rows)
        nBK    = length(B_cols)
        nnotCR = length(notC_rows)
        nnotBK = length(notB_cols)
        mat1 = [A1                  A2             zeros(Int,nnotR,1)  zeros(Int,nnotR,1)
                A3                  A4             ones(Int,nR,1)      ones(Int,nR,1)
                zeros(Int,1,nnotK)  ones(Int,1,nK) 0                   ε₂               ]
        mat2 = [ε₁                   zeros(Int,1,nBK)      ones(Int,1,nnotBK)    0
                ones(Int,nCR,1)      ones(Int,nCR,1)       D1                    D2
                zeros(Int,nnotCR,1)  zeros(Int,nnotCR,1)   D3                    D4   ]
        return _is_tu_recursive(mat1, depth+1, seen) &&
               _is_tu_recursive(mat2, depth+1, seen)

    elseif rB == 2 && rC == 0
        pivot_pos = findfirst(!iszero, B)
        pivot_pos === nothing && error("B has rank 2 but no nonzero entries")
        pi, pj = pivot_pos[1], pivot_pos[2]
        rA = size(A, 1)
        cA = size(A, 2)
        row_order = [pi; [i for i in 1:rA if i != pi]; collect(rA+1:rA+size(D,1))]
        col_order = [cA+pj; collect(1:cA); [cA+j for j in 1:size(B,2) if j != pj]]
        M_full = [A B; zeros(Int,size(D,1),cA) D]
        M_perm = M_full[row_order, col_order]
        ok, M_prime = _reduce(pivot(M_perm, 1))
        ok || return false
        return _is_tu_recursive(M_prime, depth+1, seen)

    elseif rB == 0 && rC == 2
        pivot_pos = findfirst(!iszero, C)
        pivot_pos === nothing && error("C has rank 2 but no nonzero entries")
        pi, pj = pivot_pos[1], pivot_pos[2]
        rA = size(A, 1)
        cA = size(A, 2)
        rC_size = size(C, 1)
        cD = size(D, 2)
        row_order = [rA+pi; collect(1:rA); [rA+i for i in 1:rC_size if i != pi]]
        col_order = [pj; [j for j in 1:cA if j != pj]; collect(cA+1:cA+cD)]
        M_full = [A zeros(Int,rA,cD); C D]
        M_perm = M_full[row_order, col_order]
        ok, M_prime = _reduce(pivot(M_perm, 1))
        ok || return false
        return _is_tu_recursive(M_prime, depth+1, seen)

    else
        error("Unexpected rank(B) + rank(C) = $(rB + rC)")
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Partition algorithm (Ghouila-Houri criterion)
# Originally a port of tuPartition / tuPartitionSubset / tuPartitionSearch from
# cmr/tu.c; the subset and sign enumerations are now interleaved into a single
# branch-and-prune search (see comments in _tu_partition), which is orders of
# magnitude faster than the plain 3^r enumeration on typical inputs.
# ──────────────────────────────────────────────────────────────────────────────

function _tu_partition(M::Matrix{Int})::Bool
    r, c = size(M)
    r > c && return _tu_partition(Matrix{Int}(M'))  # work over smaller dimension

    # Build CSR sparse row structure (mirrors the C CMR implementation).
    # 3 flat allocations replace the O(r) vector-of-vectors approach, improving
    # cache locality for the hot inner loops.
    row_ptr = zeros(Int, r + 1)
    for i in 1:r, j in 1:c; iszero(M[i, j]) || (row_ptr[i + 1] += 1); end
    for i in 1:r; row_ptr[i + 1] += row_ptr[i]; end   # prefix-sum → row boundaries
    nnz = row_ptr[r + 1]
    row_col = Vector{Int}(undef, nnz)   # column index of each nonzero
    row_val = Vector{Int}(undef, nnz)   # value of each nonzero
    for i in 1:r
        ptr = row_ptr[i]
        for j in 1:c
            v = M[i, j]; iszero(v) && continue
            ptr += 1; row_col[ptr] = j; row_val[ptr] = v
        end
    end

    # Ghouila-Houri: TU ⟺ every row subset R admits a signing with all
    # |column sums| ≤ 1. The subset choice and the sign search must stay
    # SEPARATE phases: each subset picks its own signs, so the ∀R and ∃signs
    # quantifiers cannot be interleaved into one in/out/± tree — subsets
    # sharing a prefix would be forced to share sign choices, giving false
    # negatives. (An interleaved variant passed 20k uniform random tests
    # before a biased fuzz caught it — beware.)
    #
    # Within the sign search for a FIXED R, two sound accelerations apply:
    #  * Branch-and-prune: proc_sum[j] = signed sum over already-signed rows,
    #    rem_nnz[j] = nonzeros of column j among not-yet-signed selected
    #    rows. Once |proc_sum[j]| > 1 + rem_nnz[j], no sign completion can
    #    bring column j back within ≤ 1 — prune the branch. At the leaves
    #    rem_nnz ≡ 0, so the invariant IS the Ghouila-Houri bound: no final
    #    column scan is needed. Only columns touched by the current row can
    #    become hopeless, keeping each node O(nnz(row)).
    #  * Sign symmetry: negating a whole signing preserves |sums|, so the
    #    first selected row takes +1 WLOG (halves the tree).
    proc_sum = zeros(Int, c)
    rem_nnz  = zeros(Int, c)     # over not-yet-signed SELECTED rows
    sel_rows = Vector{Int}(undef, r)
    ns = 0

    # Sign search for sel_rows[k..ns]; proc_sum/rem_nnz reflect rows < k signed.
    function search(k::Int)::Bool
        k > ns && return true
        row = sel_rows[k]
        lo, hi = row_ptr[row] + 1, row_ptr[row + 1]
        @inbounds for kk in lo:hi; rem_nnz[row_col[kk]] -= 1; end

        good = true                                             # sign +1
        @inbounds for kk in lo:hi
            j = row_col[kk]
            proc_sum[j] += row_val[kk]
            abs(proc_sum[j]) > 1 + rem_nnz[j] && (good = false)
        end
        found = good && search(k + 1)
        @inbounds for kk in lo:hi; proc_sum[row_col[kk]] -= row_val[kk]; end

        if !found && k > 1                                      # sign −1 (skip for first: +1 WLOG)
            good = true
            @inbounds for kk in lo:hi
                j = row_col[kk]
                proc_sum[j] -= row_val[kk]
                abs(proc_sum[j]) > 1 + rem_nnz[j] && (good = false)
            end
            found = good && search(k + 1)
            @inbounds for kk in lo:hi; proc_sum[row_col[kk]] += row_val[kk]; end
        end

        @inbounds for kk in lo:hi; rem_nnz[row_col[kk]] += 1; end
        return found
    end

    # Enumerate all 2^r subsets R (exclude-first: small violators found early);
    # rem_nnz is maintained incrementally as rows join the subset.
    function enum_subsets(row::Int)::Bool
        row > r && return search(1)
        enum_subsets(row + 1) || return false                   # exclude
        ns += 1; sel_rows[ns] = row                             # include
        for k in row_ptr[row]+1:row_ptr[row+1]
            @inbounds rem_nnz[row_col[k]] += 1; end
        result = enum_subsets(row + 1)
        for k in row_ptr[row]+1:row_ptr[row+1]
            @inbounds rem_nnz[row_col[k]] -= 1; end
        ns -= 1
        result
    end

    enum_subsets(1)
end

# ──────────────────────────────────────────────────────────────────────────────
# CMR Eulerian algorithm
# Ports tuEulerian / tuEulerianRows / tuEulerianColumns from cmr/tu.c.
# M is TU iff every square Eulerian submatrix has sum ≡ 0 (mod 4).
# A k×k submatrix is Eulerian when every row and every column within it has an
# even number of nonzeros.
# ──────────────────────────────────────────────────────────────────────────────

function _tu_eulerian(M::Matrix{Int}, max_k::Int = typemax(Int))::Bool
    r, c = size(M)
    r > c && return _tu_eulerian(Matrix{Int}(M'), max_k)

    # Build CSR sparse row structure (column indices only — no values needed here).
    # Iterating only over nonzeros mirrors the CSR format used by C CMR.
    row_ptr2 = zeros(Int, r + 1)
    for i in 1:r, j in 1:c; iszero(M[i, j]) || (row_ptr2[i + 1] += 1); end
    for i in 1:r; row_ptr2[i + 1] += row_ptr2[i]; end
    nnz2 = row_ptr2[r + 1]
    row_col2 = Vector{Int}(undef, nnz2)
    for i in 1:r
        ptr = row_ptr2[i]
        for j in 1:c; iszero(M[i, j]) && continue; ptr += 1; row_col2[ptr] = j; end
    end

    col_nz   = zeros(Int, c)    # nonzeros per column in currently selected rows
    row_nz   = zeros(Int, r)    # nonzeros per row in currently selected columns
    sub_rows = zeros(Int, r)    # sub_rows[1..k] = selected row indices
    use_cols = zeros(Int, c)    # usable column indices (even col_nz)
    col_sel  = zeros(Int, c)    # col_sel[1..k] = index-into-use_cols of chosen col
    sum_ent  = Ref(0)           # sum of entries in selected k×k submatrix

    # Pick k−n_sel more columns from use_cols[1..n_use]; n_sel already chosen.
    function enum_cols(k::Int, n_sel::Int, n_use::Int)::Bool
        if n_sel < k
            first = n_sel == 0 ? 1 : col_sel[n_sel] + 1
            last  = n_use - (k - n_sel) + 1
            for u in first:last
                col = use_cols[u]
                for s in 1:k                                    # update row_nz / sum
                    v = M[sub_rows[s], col]
                    if v != 0; sum_ent[] += v; row_nz[sub_rows[s]] += 1; end
                end
                col_sel[n_sel + 1] = u
                enum_cols(k, n_sel + 1, n_use) || return false
                for s in 1:k                                    # restore
                    v = M[sub_rows[s], col]
                    if v != 0; sum_ent[] -= v; row_nz[sub_rows[s]] -= 1; end
                end
            end
            return true
        else
            # Columns are Eulerian by construction (selected from use_cols).
            # Check whether rows are also Eulerian and sum ≢ 0 mod 4.
            sum_ent[] % 4 == 0 && return true
            for s in 1:k
                row_nz[sub_rows[s]] % 2 == 0 || return true    # row not Eulerian → ok
            end
            return false                                        # Eulerian + sum ≢ 0 mod 4
        end
    end

    # Pick k−n_sel more rows from 1..r; n_sel already chosen.
    function enum_rows(k::Int, n_sel::Int)::Bool
        if n_sel < k
            first = n_sel == 0 ? 1 : sub_rows[n_sel] + 1
            last  = r - (k - n_sel) + 1
            for row in first:last
                sub_rows[n_sel + 1] = row
                for k2 in row_ptr2[row]+1:row_ptr2[row+1]       # sparse update
                    @inbounds col_nz[row_col2[k2]] += 1; end
                enum_rows(k, n_sel + 1) || return false
                for k2 in row_ptr2[row]+1:row_ptr2[row+1]       # sparse restore
                    @inbounds col_nz[row_col2[k2]] -= 1; end
            end
            return true
        else
            # k rows chosen. Usable columns = those with even nonzero count.
            n_use = 0
            for j in 1:c
                if col_nz[j] % 2 == 0; n_use += 1; use_cols[n_use] = j; end
            end
            n_use < k && return true                            # too few usable cols
            enum_cols(k, 0, n_use)
        end
    end

    for k in 2:min(min(r, c), max_k)
        enum_rows(k, 0) || return false
    end
    true
end

# ──────────────────────────────────────────────────────────────────────────────
# Public dispatcher — mirrors CMRtuTest
# ──────────────────────────────────────────────────────────────────────────────

"""
    cmr_is_totally_unimodular(M; algorithm=:decomposition)

Test whether integer matrix `M` is totally unimodular using one of the three
algorithms from the CMR library (`src/cmr/tu.c`, `CMRtuTest`):

| `algorithm`      | CMR constant                     | Description                       |
|:-----------------|:---------------------------------|:----------------------------------|
| `:decomposition` | `CMR_TU_ALGORITHM_DECOMPOSITION` | Seymour decomposition (default)   |
| `:eulerian`      | `CMR_TU_ALGORITHM_EULERIAN`      | Eulerian submatrix criterion      |
| `:partition`     | `CMR_TU_ALGORITHM_PARTITION`     | Ghouila-Houri partition criterion |

**`:decomposition`** delegates to [`is_totally_unimodular`](@ref).

**`:eulerian`** — M is TU iff every square Eulerian submatrix (each row and column
within it has an even number of nonzeros) has total entry sum ≡ 0 (mod 4).

**`:partition`** (Ghouila-Houri) — M is TU iff for every subset R of rows there
exists a partition R = R⁺ ∪ R⁻ with |∑_{i∈R⁺} Mᵢⱼ − ∑_{i∈R⁻} Mᵢⱼ| ≤ 1 for
all columns j.

Both `:eulerian` and `:partition` are exponential-time but exact for {-1,0,1}
inputs; CMR uses them as cross-checks and for small matrices.

# Arguments
- `M::Matrix{Int}`: Integer matrix whose entries must be in {-1, 0, 1}.
- `algorithm::Symbol`: Which algorithm to use (default `:decomposition`).
"""
function cmr_is_totally_unimodular(M::Matrix{Int};
                                   algorithm::Symbol = :decomposition)::Bool
    all(m -> m in (-1, 0, 1), M) || return false
    if algorithm === :decomposition
        return is_totally_unimodular(M)
    elseif algorithm === :eulerian
        return _tu_eulerian(M)
    elseif algorithm === :partition
        return _tu_partition(M)
    else
        throw(ArgumentError("Unknown algorithm $(repr(algorithm)). " *
                            "Use :decomposition, :eulerian, or :partition."))
    end
end


end # module TotalUnimodularity
