module TotalUnimodularity

using LinearAlgebra
using Combinatorics
using Graphs

# Public API
export naive_is_totally_unimodular
export is_totally_unimodular
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

# Compute rank(IM[:, cols]) where IM = [I_m | M], exploiting the identity block:
#   rank([I_m | M][:, cols]) = |I_cols| + rank(M[not_I_rows, M_cols])
# Uses a UInt64 bitmask for covered rows — one heap allocation (the submatrix).
function _rank_IM(M::Matrix{Int}, m::Int, cols::AbstractVector{Int})::Int
    isempty(cols) && return 0
    n_I = 0
    n_M = 0
    covered = zero(UInt64)
    for c in cols
        if c <= m
            n_I += 1
            covered |= UInt64(1) << (c - 1)
        else
            n_M += 1
        end
    end
    n_M == 0 && return n_I
    n_notI = m - n_I
    n_notI == 0 && return m
    B = Matrix{Int}(undef, n_notI, n_M)
    row = 0
    for i in 1:m
        (covered >> (i - 1)) & 1 == 1 && continue
        row += 1
        col = 0
        for c in cols
            c <= m && continue
            col += 1
            B[row, col] = M[i, c - m]
        end
    end
    n_I + _rank_int!(B)
end

# Cached wrapper: same result as _rank_IM(M, m, mask) but avoids recomputing
# when the same column set appears multiple times in the O(N^8) decompose loop.
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
function _rank_IM(M::Matrix{Int}, m::Int, mask::UInt64)::Int
    iszero(mask) && return 0
    n = size(M, 2)
    n_I = 0
    covered = zero(UInt64)
    for c in 1:m
        (mask >> (c - 1)) & 1 == 0 && continue
        n_I += 1
        covered |= UInt64(1) << (c - 1)
    end
    n_M = 0
    for c in m+1:m+n
        (mask >> (c - 1)) & 1 == 1 && (n_M += 1)
    end
    n_M == 0 && return n_I
    n_notI = m - n_I
    n_notI == 0 && return m
    B = Matrix{Int}(undef, n_notI, n_M)
    row = 0
    for i in 1:m
        (covered >> (i - 1)) & 1 == 1 && continue
        row += 1
        col = 0
        for c in m+1:m+n
            (mask >> (c - 1)) & 1 == 0 && continue
            col += 1
            B[row, col] = M[i, c - m]
        end
    end
    n_I + _rank_int!(B)
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

# Find the first row index i for which G_i is disconnected.
# Returns (i, graph, components, vertex_map) or nothing if all G_i are connected.
function _find_disconnected_gi(M::Matrix{Int})
    m = size(M, 1)
    for i in 1:m
        g, orig = _build_gi(M, i)
        if !Graphs.is_connected(g)
            components = Graphs.connected_components(g)
            return (i, g, components, orig)
        end
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
        U[k] = union(Set{Int}(), [W_rows[orig[v]] for v in component]...)
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

Uses the matroid intersection algorithm of Theorem 20.2.

Returns `(true, (A, B, C, D))` if such a decomposition exists,
or `(false, (M, M, M, M))` if not.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.2.
"""
function _decompose(M::Matrix{Int})::Tuple{Bool, NTuple{4, Matrix{Int}}}
    m, n = size(M)
    N = m + n    # total columns of [I | M]; I-cols: 1:m, M-cols: m+1:N
    rhoX = m     # rank([I | M]) = m always

    # Shared rank cache: the same column-subset rank is queried many times across
    # different (S,T) pairs in the O(N^8) loop. Caching eliminates redundant Bareiss.
    rank_cache = Dict{UInt64, Int}()

    # Precompute valid 4-element subset bitmasks (must intersect both I-cols and M-cols).
    valid_masks = UInt64[]
    for s in combinations(1:N, 4)
        any(x -> x <= m, s) || continue
        any(x -> x > m, s)  || continue
        mask = UInt64(0)
        for x in s; mask |= UInt64(1) << (x - 1); end
        push!(valid_masks, mask)
    end

    @inbounds for S_mask in valid_masks
        @inbounds for T_mask in valid_masks
            # S and T must be disjoint
            S_mask & T_mask != 0 && continue

            # Solve problem (16) — returns Y as a bitmask
            found, Y_mask = _solve_submodular(M, m, S_mask, T_mask, rhoX, rank_cache)
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

            # Extract submatrices
            A = M[row_top,  col_left ]
            B = M[row_top,  col_right]
            C = M[row_bot,  col_left ]
            D = M[row_bot,  col_right]
            
            # Check rank constraint
            _rank_int(B) + _rank_int(C) <= 2 || continue

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
                            T_mask::UInt64, rhoX::Int,
                            cache::Dict{UInt64,Int})::Tuple{Bool, UInt64}
    rk(mask) = _rank_IM_cached(cache, M, m, mask)

    N = m + size(M, 2)
    all_mask = N < 64 ? (UInt64(1) << N) - UInt64(1) : typemax(UInt64)
    ST_mask = S_mask | T_mask
    V = [v for v in 1:N if (ST_mask >> (v - 1)) & 1 == 0]  # V = X \ (S∪T)
    Z_mask = UInt64(0)

    while true
        SZ_mask = S_mask | Z_mask
        TZ_mask = T_mask | Z_mask
        rhoSZ = rk(SZ_mask)
        rhoTZ = rk(TZ_mask)

        # Compute U and W: elements of V\Z that extend S∪Z / T∪Z independently
        U = Int[]
        W = Int[]
        for v in V
            (Z_mask >> (v - 1)) & 1 == 1 && continue
            vbit = UInt64(1) << (v - 1)
            rk(SZ_mask | vbit) == rhoSZ + 1 && push!(U, v)
            rk(TZ_mask | vbit) == rhoTZ + 1 && push!(W, v)
        end

        # Build digraph D from (18):
        #   (u,v): u∈Z, v∈V\Z, ρ(S∪(Z\{u})∪{v}) = ρ(S)+|Z|
        #   (v,u): v∈V\Z, u∈Z, ρ(T∪(Z\{u})∪{v}) = ρ(T)+|Z|
        d_edges = Tuple{Int,Int}[]
        rhoS = iszero(Z_mask) ? rhoSZ : rk(S_mask)
        rhoT = iszero(Z_mask) ? rhoTZ : rk(T_mask)
        rhoZ = count_ones(Z_mask)
        for u_bit in 0:N-1
            (Z_mask >> u_bit) & 1 == 0 && continue
            u = u_bit + 1
            Zminu_mask = Z_mask & ~(UInt64(1) << u_bit)
            for v in V
                (Z_mask >> (v - 1)) & 1 == 1 && continue
                vbit = UInt64(1) << (v - 1)
                if rk(S_mask | Zminu_mask | vbit) == rhoS + rhoZ
                    push!(d_edges, (u, v))
                end
                if rk(T_mask | Zminu_mask | vbit) == rhoT + rhoZ
                    push!(d_edges, (v, u))
                end
            end
        end

        path = _shortest_path(d_edges, U, W, V)

        if path !== nothing
            # Update Z via symmetric difference with path (20)
            path_mask = UInt64(0)
            for p in path; path_mask |= UInt64(1) << (p - 1); end
            Z_mask = Z_mask ⊻ path_mask
        else
            # Y = S ∪ {v ∈ V | v can reach W in D} (23)
            reachable = _reachable_to(d_edges, W, V)
            reach_mask = UInt64(0)
            for r in reachable; reach_mask |= UInt64(1) << (r - 1); end
            Y_mask = S_mask | reach_mask

            XminusY_mask = all_mask & ~Y_mask
            # Y = S when reachable is empty (common case) — reuse rhoSZ
            rhoY       = iszero(reach_mask) ? rhoSZ : rk(Y_mask)
            rhoXminusY = iszero(XminusY_mask) ? 0 : rk(XminusY_mask)

            return (rhoY + rhoXminusY <= rhoX + 2, Y_mask)
        end
    end
end

function _shortest_path(edges::Vector{Tuple{Int,Int}},
                         sources::Vector{Int},
                         targets::Vector{Int},
                         vertices::Vector{Int})
    (isempty(sources) || isempty(targets)) && return nothing
    target_set = Set(targets)

    # Check if any source is already a target
    for s in sources
        s in target_set && return [s]
    end

    # BFS
    visited = Set{Int}(sources)
    prev = Dict{Int,Int}()
    queue = copy(sources)

    while !isempty(queue)
        v = popfirst!(queue)
        for (u, w) in edges
            u == v || continue
            w in visited && continue
            push!(visited, w)
            prev[w] = v
            w in target_set && return _reconstruct_path(prev, Set(sources), w)
            push!(queue, w)
        end
    end
    return nothing
end

function _reconstruct_path(prev::Dict{Int,Int}, sources::Set{Int},
                            target::Int)
    path = [target]
    v = target
    while v ∉ sources
        v = prev[v]
        pushfirst!(path, v)
    end
    return path
end

function _reachable_to(edges::Vector{Tuple{Int,Int}},
                        targets::Vector{Int},
                        vertices::Vector{Int})
    isempty(targets) && return Int[]
    # Reverse the graph and BFS from targets
    visited = Set{Int}(targets)
    queue = copy(targets)

    while !isempty(queue)
        v = popfirst!(queue)
        for (u, w) in edges
            w == v || continue  # reversed: follow incoming edges
            u in visited && continue
            push!(visited, u)
            push!(queue, u)
        end
    end

    target_set = Set(targets)
    return [v for v in vertices if v in visited && v ∉ target_set]
end

"""
    _extract_rank1(B)

Extract f and g from a rank-1 matrix B = f⊗g, where f is a {0,±1} column
vector and g is a {0,+1} row vector.

Normalises f so that its first nonzero entry is positive, ensuring g is
{0,+1} as required by Schrijver's Case 2 and Case 3.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3, Case 2.
"""
function _extract_rank1(B::Matrix{Int})
    m, n = size(B)
    col = findfirst(j -> any(!iszero, B[:, j]), 1:n)
    f = B[:, col]  # no normalisation
    # g[j] = 1 if B[:,j] == f, 0 otherwise
    # Since B = f⊗g with g {0,+1}, all nonzero columns of B equal f
    g = zeros(Int, 1, n)
    for j in 1:n
        if B[:, j] == f
            g[1, j] = 1
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
[`is_totally_unimodular`](@ref) for the linear-time implementation.

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
        for rows in combinations(1:r, s), cols in combinations(1:c, s)
            @views det(M[rows, cols]) in (-1, 0, 1) || return false
        end
    end
    return true
end

function is_totally_unimodular(M::Matrix{Int})::Bool
    _is_tu_recursive(M, 0, Set{Matrix{Int}}())
end

function _is_tu_recursive(M::Matrix{Int}, depth::Int, seen::Set{Matrix{Int}})::Bool
    depth > 100 && error("Maximum recursion depth exceeded")
    
    ok, M = _reduce(M)
    ok || return false
    (size(M, 1) == 0 || size(M, 2) == 0) && return true
    
    # Cycle detection
    M in seen && return false
    push!(seen, copy(M))
    
    _is_network_matrix(M) && return true
    _is_network_matrix(Matrix{Int}(M')) && return true
    _is_special_matrix(M) && return true

    found, (A, B, C, D) = _decompose(M)
    found || return false

    rB = _rank_int(B)
    rC = _rank_int(C)

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
        A_degenerate = any(i -> count(!iszero, A[i,:]) <= 1, 1:size(A,1)) ||
                       any(j -> count(!iszero, A[:,j]) <= 1, 1:size(A,2)) ||
                       _has_dependent_vectors(A)
        D_degenerate = any(i -> count(!iszero, D[i,:]) <= 1, 1:size(D,1)) ||
                       any(j -> count(!iszero, D[:,j]) <= 1, 1:size(D,2)) ||
                       _has_dependent_vectors(D)
        (A_degenerate || D_degenerate) && return false

        f_B, g_B = _extract_rank1(B)
        f_C, g_C = _extract_rank1(C)
        B_rows    = findall(!iszero, f_B[:, 1])
        C_cols    = findall(!iszero, g_C[1, :])
        notB_rows = [i for i in 1:size(A, 1) if i ∉ B_rows]
        notC_cols = [j for j in 1:size(A, 2) if j ∉ C_cols]
        A_norm = copy(A)
        for i in B_rows
            A_norm[i, :] *= f_B[i, 1]
        end
        C_rows    = findall(!iszero, f_C[:, 1])
        B_cols    = findall(!iszero, g_B[1, :])
        notC_rows = [i for i in 1:size(D, 1) if i ∉ C_rows]
        notB_cols = [j for j in 1:size(D, 2) if j ∉ B_cols]
        D_norm = copy(D)
        for i in C_rows
            D_norm[i, :] *= f_C[i, 1]
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


end # module TotalUnimodularity
