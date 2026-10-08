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
    all(mask) && return M
    return dim == 1 ? M[mask, :] : M[:, mask]
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
    # (sort on a Vector: sort(::NTuple) needs Julia ≥ 1.12.)
    row_abs_sums = sort!([sum(abs, @view M[i,:]) for i in 1:5])
    if row_abs_sums == [3,3,3,3,3]
        _is_sign_and_permutation_equivalent(M, F_1) && return true
    end
    if row_abs_sums == [3,3,3,3,5]
        _is_sign_and_permutation_equivalent(M, F_2) && return true
    end
    return false
end

# Mark the first vector of each class of equal-or-opposite vectors of M along
# `dim` (1: rows, 2: columns). Each vector is hashed with the sign that makes
# its first nonzero positive, so a class shares one hash; vectors with the
# same hash are chained and compared entry by entry. The cost is linear in
# the size of M, where comparing all pairs is quadratic in the vector count.
function _independent_mask(M::Matrix{Int}, dim::Int)::BitVector
    n = size(M, dim)
    len = size(M, 3 - dim)
    keep = trues(n)
    n < 2 && return keep
    entry(i, k) = dim == 1 ? M[i, k] : M[k, i]
    sign = ones(Int, n)
    first_with = Dict{UInt, Int}()       # hash => first kept vector with it
    next_with = zeros(Int, n)            # chain of kept vectors sharing a hash
    sizehint!(first_with, n)
    @inbounds for i in 1:n
        h = zero(UInt)
        s = 0
        for k in 1:len
            x = entry(i, k)
            s == 0 && x != 0 && (s = x)
            h = hash(s * x, h)
        end
        s != 0 && (sign[i] = s)
        j = get(first_with, h, 0)
        if j == 0
            first_with[h] = i
            continue
        end
        last = j
        while j != 0
            if all(k -> sign[i] * entry(i, k) == sign[j] * entry(j, k), 1:len)
                keep[i] = false
                break
            end
            last = j
            j = next_with[j]
        end
        keep[i] && (next_with[last] = i)
    end
    keep
end

# Remove all but the first row of each class of equal-or-opposite rows.
function _drop_dependent_rows(M::Matrix{Int})
    keep = _independent_mask(M, 1)
    all(keep) ? M : M[keep, :]
end

# Remove all but the first column of each class of equal-or-opposite columns.
function _drop_dependent_cols(M::Matrix{Int})
    keep = _independent_mask(M, 2)
    all(keep) ? M : M[:, keep]
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
#
# Works on the leading m×n block of a pre-allocated scratch buffer, so no heap
# allocation.
function _rank_float_view!(B::Matrix{Float64}, m::Int, n::Int)::Int
    r = 0
    @inbounds for col in 1:n
        prow = 0; best = 1e-7  # see threshold note above
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
        size(N) == size(M) && return (true, M)     # nothing was dropped
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
    all(j -> count(!iszero, @view M[:, j]) <= 2, 1:size(M, 2))

# Case 1: test if M is a network matrix when all columns have ≤2 nonzeros.
# Let G be the graph on the rows in which a column with two nonzeros of the
# same sign is an edge between its two rows, and one with opposite signs a
# path of length 2 through a new vertex. M is a network matrix iff G is
# bipartite, i.e. iff the rows can be 2-coloured so that same-sign columns
# join different colours and opposite-sign columns equal ones. That is
# checked with a union-find structure carrying each row's colour relative to
# its root, without building G.
function _is_network_matrix_few_nonzeros(M::Matrix{Int})
    m, n = size(M)
    parent = collect(1:m)
    parity = zeros(Int, m)               # colour relative to parent
    function find(x)
        root = x; p = 0
        @inbounds while parent[root] != root
            p ⊻= parity[root]; root = parent[root]
        end
        @inbounds while parent[x] != root && x != root   # path compression
            nxt = parent[x]; q = parity[x]
            parent[x] = root; parity[x] = p
            p ⊻= q; x = nxt
        end
        root
    end
    @inbounds for j in 1:n
        i = 0; k = 0; cnt = 0
        for r in 1:m
            M[r, j] == 0 && continue
            cnt += 1
            cnt == 1 ? (i = r) : (k = r)
        end
        cnt == 2 || continue
        differ = M[i, j] == M[k, j] ? 1 : 0
        ri = find(i); rk = find(k)
        ci = parent[i] == i ? 0 : parity[i]      # relative to root, after compression
        ck = parent[k] == k ? 0 : parity[k]
        if ri == rk
            ci ⊻ ck == differ || return false
        else
            parent[ri] = rk
            parity[ri] = ci ⊻ ck ⊻ differ
        end
    end
    return true
end

# Find the first row index i for which G_i is disconnected, where G_i is the
# graph on the rows other than i in which two rows are adjacent when they
# share a column that is zero in row i.
# Returns (i, components, orig) or nothing if all G_i are connected. The
# vertices of G_i are numbered 1..m-1 in row order, orig[v] is the row of
# vertex v, and each component is a vector of vertices.
#
# G_i is never built: its components are found by a search that alternates
# between rows and the columns that are zero in row i, on the supports of M.
# Each column is expanded once, so one i costs O(nnz).
function _find_disconnected_gi(M::Matrix{Int})
    m, n = size(M)
    m < 3 && return nothing                  # G_i has at most one vertex
    row_cols = [findall(!iszero, @view M[r, :]) for r in 1:m]
    col_rows = [findall(!iszero, @view M[:, c]) for c in 1:n]
    seen_row = falses(m)
    seen_col = falses(n)
    queue = Vector{Int}(undef, m)

    # Search from row `start`, appending the rows reached to queue[tail+1:end];
    # returns the new tail.
    function component!(i, start, tail)
        head = tail
        seen_row[start] = true
        queue[tail += 1] = start
        @inbounds while head < tail
            r = queue[head += 1]
            for c in row_cols[r]
                (seen_col[c] || M[i, c] != 0) && continue
                seen_col[c] = true
                for r2 in col_rows[c]
                    seen_row[r2] && continue
                    seen_row[r2] = true
                    queue[tail += 1] = r2
                end
            end
        end
        tail
    end

    for i in 1:m
        fill!(seen_row, false)
        fill!(seen_col, false)
        seen_row[i] = true
        tail = component!(i, i == 1 ? 2 : 1, 0)
        tail == m - 1 && continue            # G_i is connected
        vertex(r) = r < i ? r : r - 1
        components = [[vertex(queue[k]) for k in 1:tail]]
        for r in 1:m
            seen_row[r] && continue
            from = tail
            tail = component!(i, r, tail)
            push!(components, [vertex(queue[k]) for k in from+1:tail])
        end
        return (i, components, [r for r in 1:m if r != i])
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
        W_rows[j] = Set(c for c in W if M[j, c] != 0)
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
                             orig::Vector{Int}; drop_zero_columns::Bool = false)
    map(components) do component
        rows = [i; [orig[v] for v in component]]
        drop_zero_columns || return M[rows, :]
        cols = [j for j in 1:size(M, 2) if any(r -> M[r, j] != 0, rows)]
        M[rows, cols]
    end
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

    i, components, orig = result
    W, W_rows, U = _compute_w_sets(M, i, components, orig)
    h = _build_h(components, orig, W_rows, U)

    # H must be bipartite
    Graphs.is_bipartite(h) || return false

    # Recursively test each submatrix. Columns that are zero on the rows of
    # a submatrix play no part in any step of the test, and dropping them
    # keeps the recursion from carrying the full width of M all the way down.
    submatrices = _split_submatrices(M, i, components, orig; drop_zero_columns = true)
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
- `M`: An integer matrix whose entries are in {-1, 0, 1}.
- `k::Int`: Size of the leading square submatrix to pivot on. An
  `ArgumentError` is thrown if `k` is out of range or the leading k×k
  submatrix does not have determinant ±1.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function pivot(M::Matrix{Int}, k::Int)
    0 <= k <= min(size(M)...) ||
        throw(ArgumentError("k = $k must be between 0 and min(size(M)) = $(min(size(M)...))."))
    abs(_det_int!(M[1:k, 1:k])) == 1 ||
        throw(ArgumentError("The leading $k×$k submatrix must have determinant ±1."))
    @views E = M[1:k,     1:k    ]
    @views B = M[k+1:end, 1:k    ]
    @views C = M[1:k,     k+1:end]
    @views D = M[k+1:end, k+1:end]
    Einv = Matrix{Int}(round.(inv(Matrix{Rational{Int}}(E))))
    [-Einv        Einv*C
      B*Einv  D - B*Einv*C]
end

pivot(M::AbstractMatrix{<:Integer}, k::Integer) = pivot(Matrix{Int}(M), Int(k))

"""
    _decompose(M)

Test whether the rows and columns of `M` can be permuted so that

    M = [A  B]
        [C  D]

with rank(B) + rank(C) ≤ 2 and both A and D having r + c ≥ 4, by
enumerating all 2^m × 2^n row/column bipartitions. `M` must be at most
12×12. Most splits fail on a GF(2) rank bound or on rank(B) alone, so the
search exits early almost everywhere.

This is the exhaustive counterpart of `_find_two_separation` and
`_find_three_separation`, used by
`cmr_is_totally_unimodular(M; algorithm = :decomposition)` as an independent
check on them.

Returns `(true, (A, B, C, D))` if such a decomposition exists,
or `(false, (M, M, M, M))` if not.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.2.
"""
function _decompose(M::Matrix{Int})::Tuple{Bool, NTuple{4, Matrix{Int}}}
    m, n = size(M)
    (m <= 12 && n <= 12) ||
        throw(ArgumentError("_decompose handles matrices up to 12×12, got $m×$n."))

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
                return (true, (M[row_top[1:nrt], col_left[1:ncl]],
                               M[row_top[1:nrt], col_right[1:ncr]],
                               M[row_bot[1:nrb], col_left[1:ncl]],
                               M[row_bot[1:nrb], col_right[1:ncr]]))
            end
        end
    end
    return (false, (M, M, M, M))
end

# Strongly connected components of the rule digraph of _find_two_separation
# for pivot (i0, j0) (Tarjan's algorithm, iterative). Nodes are the rows i ≠ i0
# (as i) and the columns j ≠ j0 (as m + j). `order` receives the nodes
# component by component, successors first, and `ends` the position in `order`
# at which each component ends — so every prefix of `order` that stops at a
# component boundary is closed under the rules.
function _two_separation_sccs!(order::Vector{Int}, ends::Vector{Int},
                               index::Vector{Int}, low::Vector{Int}, pos::Vector{Int},
                               onstack::BitVector, stack::Vector{Int}, calls::Vector{Int},
                               M::Matrix{Int}, i0::Int, j0::Int)
    m, n = size(M)
    p = M[i0, j0]
    fill!(index, 0)
    empty!(order); empty!(ends); empty!(stack); empty!(calls)
    counter = 0
    @inbounds for root in 1:m+n
        (root == i0 || root == m + j0 || index[root] != 0) && continue
        index[root] = low[root] = (counter += 1)
        pos[root] = 0
        push!(stack, root); onstack[root] = true
        push!(calls, root)
        while !isempty(calls)
            v = calls[end]
            descended = false
            if v <= m
                # minor rule: row v pulls in column w
                q = M[v, j0]
                while pos[v] < n
                    w = (pos[v] += 1)
                    (w == j0 || p * M[v, w] == M[i0, w] * q) && continue
                    u = m + w
                    if index[u] == 0
                        index[u] = low[u] = (counter += 1)
                        pos[u] = 0
                        push!(stack, u); onstack[u] = true
                        push!(calls, u)
                        descended = true
                        break
                    elseif onstack[u]
                        low[v] = min(low[v], index[u])
                    end
                end
            else
                # support rule: column v - m pulls in row w
                c = v - m
                while pos[v] < m
                    w = (pos[v] += 1)
                    (w == i0 || iszero(M[w, c])) && continue
                    if index[w] == 0
                        index[w] = low[w] = (counter += 1)
                        pos[w] = 0
                        push!(stack, w); onstack[w] = true
                        push!(calls, w)
                        descended = true
                        break
                    elseif onstack[w]
                        low[v] = min(low[v], index[w])
                    end
                end
            end
            descended && continue
            pop!(calls)
            if low[v] == index[v]
                while true
                    u = pop!(stack)
                    onstack[u] = false
                    push!(order, u)
                    u == v && break
                end
                push!(ends, length(order))
            end
            isempty(calls) || (low[calls[end]] = min(low[calls[end]], low[v]))
        end
    end
    nothing
end

# Candidate pivots for _find_two_separation: the edges of a spanning tree of
# the support graph of M (rows and columns as vertices, nonzeros as edges).
# For a connected M, the two sides of any 2-separation are joined by a tree
# edge, and that edge is a nonzero of the rank-1 block because the other
# cross block is zero. If M is not connected, every nonzero is returned.
function _two_separation_pivots(M::Matrix{Int})::Vector{Tuple{Int,Int}}
    m, n = size(M)
    pivots = Tuple{Int,Int}[]
    seenR = falses(m)
    seenC = falses(n)
    queue = [1]                          # rows as i, columns as m + j
    seenR[1] = true
    head = 0
    @inbounds while head < length(queue)
        v = queue[head += 1]
        if v <= m
            for c in 1:n
                (seenC[c] || iszero(M[v, c])) && continue
                seenC[c] = true; push!(queue, m + c); push!(pivots, (v, c))
            end
        else
            c = v - m
            for r in 1:m
                (seenR[r] || iszero(M[r, c])) && continue
                seenR[r] = true; push!(queue, r); push!(pivots, (r, c))
            end
        end
    end
    length(queue) == m + n && return pivots
    [(i, j) for j in 1:n for i in 1:m if !iszero(M[i, j])]
end

"""
    _find_two_separation(M)

Search for a 2-separation of `M`: a split of the rows into R1 ∪ R2 and the
columns into C1 ∪ C2, each side holding at least two rows-plus-columns, with

    M[R2, C1] = 0   and   rank M[R1, C2] = 1.

Returns `(R1, C1)` as sorted index vectors, or `nothing` if `M` has no
2-separation. `M` must have entries in {-1, 0, 1}. Every 2-separation of a
connected matrix has exactly one nonzero cross block, so naming the sides so
that it is `M[R1, C2]` loses no generality.

Polynomial, O((m + n) · m · n) for a connected matrix. Fix a nonzero entry
(i0, j0) of the rank-1 block: row i0 is on side 1, column j0 on side 2.
Membership of side 1 is then closed under two single-premise rules:

  * a column on side 1 pulls in every row where it is nonzero (keeps
    M[R2, C1] = 0);
  * a row r on side 1 pulls in every column c for which the 2×2 minor on
    rows {i0, r}, columns {j0, c} is nonzero (a block with M[i0, j0] ≠ 0 has
    rank 1 iff all 2×2 minors through that entry vanish).

So the rules form a digraph on the other m+n-2 rows and columns, and the
valid choices of side 1 are exactly {i0} ∪ S for S a nonempty proper subset
closed under its edges. Such an S exists iff the digraph has more than one
strongly connected component. Only the edges of a spanning tree of the
support graph need to be tried as (i0, j0), see `_two_separation_pivots`.

Balanced splits are preferred, because they keep the recursion on the two
pieces shallow, where always peeling off a small piece would re-examine the
large remainder once per piece: for each pivot, the closed set chosen is the
union of components, in the order Tarjan's algorithm emits them, that is
nearest to half of the rows and columns, and a few pivots are compared
before settling on one.
"""
function _find_two_separation(M::Matrix{Int})::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}}
    m, n = size(M)
    (m + n < 4 || m < 1 || n < 1) && return nothing
    n_nodes = m + n - 2
    order = Int[]; ends = Int[]; stack = Int[]; calls = Int[]
    sizehint!(order, n_nodes)
    index = Vector{Int}(undef, m + n)
    low = Vector{Int}(undef, m + n)
    pos = Vector{Int}(undef, m + n)
    onstack = falses(m + n)

    # Most balanced closed set for one pivot: (size, R1, C1), size 0 if none.
    function split(i0, j0)
        _two_separation_sccs!(order, ends, index, low, pos, onstack, stack, calls, M, i0, j0)
        length(ends) == 1 && return (0, Int[], Int[])       # strongly connected
        best = ends[1]
        for e in @view ends[1:end-1]
            abs(2e - n_nodes) < abs(2best - n_nodes) && (best = e)
        end
        R1 = [i0]; C1 = Int[]
        for v in @view order[1:best]
            v <= m ? push!(R1, v) : push!(C1, v - m)
        end
        (min(best, n_nodes - best), sort!(R1), sort!(C1))
    end

    # How balanced the split can be depends on the pivot: one inside a small
    # piece at the end of a long chain can only cut that piece off. So a few
    # pivots spread over the spanning tree are tried first and the best split
    # kept, stopping early at one with a quarter of the matrix on each side.
    pivots = _two_separation_pivots(M)
    L = length(pivots)
    L == 0 && return nothing                    # zero matrix
    tried = falses(L)
    found = (0, Int[], Int[])
    for f in (4, 2, 6, 1, 5, 3, 7)
        k = clamp((f * L) ÷ 8, 1, L)
        tried[k] && continue
        tried[k] = true
        cand = split(pivots[k]...)
        cand[1] > found[1] && (found = cand)
        4 * found[1] >= n_nodes && break
    end
    found[1] > 0 && return (found[2], found[3])
    for k in 1:L
        tried[k] && continue
        cand = split(pivots[k]...)
        cand[1] > 0 && return (cand[2], cand[3])
    end
    return nothing
end

# Workspace for _rule_sccs!.
struct _SccWork
    order::Vector{Int}
    ends::Vector{Int}
    index::Vector{Int}
    low::Vector{Int}
    pos::Vector{Int}
    onstack::BitVector
    stack::Vector{Int}
    calls::Vector{Int}
end
_SccWork(N::Int) = _SccWork(Int[], Int[], Vector{Int}(undef, N), Vector{Int}(undef, N),
                            Vector{Int}(undef, N), falses(N), Int[], Int[])

# Strongly connected components (Tarjan's algorithm, iterative) of a digraph
# on the rows (as i) and columns (as m + j) of an m×n matrix, in which every
# edge joins a row to a column or a column to a row: `rowedge(r, c)` says
# whether row r points to column c, `coledge(c, r)` whether column c points
# to row r. Nodes with `skip[v]` set are left out. On return `w.order` holds
# the nodes component by component, successors first, and `w.ends` the
# position in `w.order` at which each component ends — so every prefix of
# `w.order` that stops at a component boundary is closed under the edges.
#
# The predicates are only evaluated where an edge is possible: row r can only
# point to the columns in `row_sup[r]` and in `extra_cols`, column c only to
# the rows in `col_sup[c]` and in `extra_rows`. For the rules used here that
# is the support of the row or column plus the supports of the fixed rows or
# columns, which keeps one pass near O(nnz) on sparse matrices.
function _rule_sccs!(w::_SccWork, m::Int, n::Int, skip::BitVector,
                     rowedge::F, coledge::G,
                     row_sup::Vector{Vector{Int}}, extra_cols::Vector{Int},
                     col_sup::Vector{Vector{Int}}, extra_rows::Vector{Int}) where {F, G}
    index, low, pos, onstack, stack, calls = w.index, w.low, w.pos, w.onstack, w.stack, w.calls
    fill!(index, 0)
    empty!(w.order); empty!(w.ends); empty!(stack); empty!(calls)
    counter = 0
    @inbounds for root in 1:m+n
        (skip[root] || index[root] != 0) && continue
        index[root] = low[root] = (counter += 1)
        pos[root] = 0
        push!(stack, root); onstack[root] = true
        push!(calls, root)
        while !isempty(calls)
            v = calls[end]
            descended = false
            own = v <= m ? row_sup[v] : col_sup[v - m]
            extra = v <= m ? extra_cols : extra_rows
            lim = length(own) + length(extra)
            while pos[v] < lim
                k = (pos[v] += 1)
                x = k <= length(own) ? own[k] : extra[k - length(own)]
                u = v <= m ? m + x : x
                skip[u] && continue
                (v <= m ? rowedge(v, x) : coledge(v - m, x)) || continue
                if index[u] == 0
                    index[u] = low[u] = (counter += 1)
                    pos[u] = 0
                    push!(stack, u); onstack[u] = true
                    push!(calls, u)
                    descended = true
                    break
                elseif onstack[u]
                    low[v] = min(low[v], index[u])
                end
            end
            descended && continue
            pop!(calls)
            if low[v] == index[v]
                while true
                    u = pop!(stack)
                    onstack[u] = false
                    push!(w.order, u)
                    u == v && break
                end
                push!(w.ends, length(w.order))
            end
            isempty(calls) || (low[calls[end]] = min(low[calls[end]], low[v]))
        end
    end
    nothing
end

# After _rule_sccs!: a set of nodes that is closed under the edges, has at
# least `lo` and at most `hi` nodes and satisfies `accept`, as a vector, or
# `nothing` if none is found. Prefixes of `w.order` are tried first, nearest
# to the middle of the range first. If no prefix fits, all unions of components are
# checked, provided there are at most 12 components: with lo = 2 and hi two
# less than the number of nodes, as _find_three_separation calls it, a miss
# means every prefix has 1 node or all but 1, so there are at most three.
function _closed_set(w::_SccWork, m::Int, n::Int, skip::BitVector, lo::Int, hi::Int,
                     rowedge::F, coledge::G, accept::H) where {F, G, H}
    order, ends = w.order, w.ends
    lo > hi && return nothing
    fits = [e for e in ends if lo <= e <= hi]
    if !isempty(fits)
        for e in sort!(fits; by = e -> abs(2e - lo - hi))
            set = order[1:e]
            accept(set) && return set
        end
        return nothing
    end

    # No prefix fits. Unions of components that are not prefixes can still be
    # closed (two components with no edge between them, say).
    k = length(ends)
    (k < 2 || k > 12) && return nothing
    comp = zeros(Int, m + n)
    let c = 1
        for (pos, v) in enumerate(order)
            pos > ends[c] && (c += 1)
            comp[v] = c
        end
    end
    sizes = [ends[c] - (c == 1 ? 0 : ends[c-1]) for c in 1:k]
    succ = zeros(UInt, k)                # components each component points into
    @inbounds for v in order
        if v <= m
            for c in 1:n
                (skip[m + c] || !rowedge(v, c)) && continue
                succ[comp[v]] |= UInt(1) << (comp[m + c] - 1)
            end
        else
            for r in 1:m
                (skip[r] || !coledge(v - m, r)) && continue
                succ[comp[v]] |= UInt(1) << (comp[r] - 1)
            end
        end
    end
    for mask in UInt(1):(UInt(1) << k) - UInt(2)
        total = 0; closed = true
        for c in 1:k
            (mask >> (c - 1)) & 1 == 1 || continue
            total += sizes[c]
            succ[c] & ~mask == 0 || (closed = false; break)
        end
        (closed && lo <= total <= hi) || continue
        set = [v for v in order if (mask >> (comp[v] - 1)) & 1 == 1]
        accept(set) && return set
    end
    nothing
end

# Spanning forest of the bipartite graph on the rows other than `i0` and the
# columns other than `j0` of an m×n matrix whose edges are the pairs (r, c)
# with `edge(r, c)`. Returns the forest edges and the component number of
# every row and column (0 for i0 and j0).
function _rule_forest(m::Int, n::Int, i0::Int, j0::Int, edge::F) where {F}
    forest = Tuple{Int,Int}[]
    rcomp = zeros(Int, m)
    ccomp = zeros(Int, n)
    queue = Int[]                        # rows as i, columns as m + j
    k = 0
    @inbounds for start in 1:m+n
        if start <= m
            (start == i0 || rcomp[start] != 0) && continue
            rcomp[start] = (k += 1)
        else
            (start - m == j0 || ccomp[start - m] != 0) && continue
            ccomp[start - m] = (k += 1)
        end
        empty!(queue); push!(queue, start)
        head = 0
        while head < length(queue)
            v = queue[head += 1]
            if v <= m
                for c in 1:n
                    (c == j0 || ccomp[c] != 0 || !edge(v, c)) && continue
                    ccomp[c] = k; push!(queue, m + c); push!(forest, (v, c))
                end
            else
                c = v - m
                for r in 1:m
                    (r == i0 || rcomp[r] != 0 || !edge(r, c)) && continue
                    rcomp[r] = k; push!(queue, r); push!(forest, (r, c))
                end
            end
        end
    end
    forest, rcomp, ccomp
end

"""
    _find_three_separation(M; accept = (R1, C1) -> true)

Search for a 3-separation of `M`: a split of the rows into R1 ∪ R2 and the
columns into C1 ∪ C2, each side holding at least four rows-plus-columns, with

    rank M[R1, C2] + rank M[R2, C1] = 2.

Returns `(R1, C1)` as sorted index vectors, or `nothing` if there is none.
`M` must have entries in {-1, 0, 1}, be connected and have no 2-separation
(so the rank sum of a split with four elements on each side is never less
than 2).

With `accept`, only splits for which `accept(R1, C1)` holds are returned.
The search then stays sound but is no longer exhaustive: for each choice of
fixed elements only some of the closed sets are offered to `accept`.

Write B = M[R1, C2] and C = M[R2, C1]. The sides can be named so that B
contains an edge (i1, j1) of a spanning tree of the support graph, because
some tree edge joins the two sides. Two kinds of split remain.

**rank B = rank C = 1.** Fix also a nonzero (i2, j2) of C. A block with a
nonzero entry has rank 1 iff every 2×2 minor through that entry vanishes, so

  * a row r on side 1 pulls in every column c whose minor with (i1, j1) is
    nonzero (keeps rank B = 1);
  * a column c on side 1 pulls in every row r whose minor with (i2, j2) is
    nonzero (keeps rank C = 1).

**rank B = 2, C = 0.** Fix also (i2, j2) with rows i1, i2 and columns j1, j2
forming a nonsingular 2×2 block of B; one exists for every nonzero (i1, j1)
of a rank-2 block. A block containing a nonsingular 2×2 block has rank 2 iff
every 3×3 minor containing it vanishes, so

  * a row r on side 1 pulls in every column c whose 3×3 minor with the block
    is nonzero (keeps rank B = 2);
  * a column c on side 1 pulls in every row where it is nonzero (keeps C = 0).

Either way the rules have a single premise each and form a digraph on the
rows and columns other than the four fixed ones; the rules never involve a
fixed row or column. The admissible sides are the two fixed elements of side
1 together with a set that is closed under the edges, holding at least 2 and
at most m + n - 6 nodes.

Not every (i2, j2) has to be tried. Take the graph on the rows other than i1
and the columns other than j1 whose edges are the nonzeros of M that cannot
lie in B (first kind: B is rank 1 through (i1, j1), so its nonzeros (r, c)
have M[r, j1] ≠ 0, M[i1, c] ≠ 0 and a vanishing minor), or the nonzeros of M
that stay nonzero in the Schur complement of (i1, j1) (second kind: these
cannot lie in C, which is zero). An edge of this graph that joins the two
sides is then a valid (i2, j2), and if one does, so does an edge of any
spanning forest. Otherwise every component lies on one side, and (i2, j2)
is one of the remaining candidates that join two components — few, as they
all sit in rows that are nonzero in column j1 and columns that are nonzero
in row i1. That leaves about m + n candidates per (i1, j1) and
O((m + n)² · m · n) work in all, where trying every candidate
(`exhaustive = true`, kept for cross-checking) is O((m + n) · (m·n)²).
"""
function _find_three_separation(M::Matrix{Int}; accept::A = (R1, C1) -> true,
                                exhaustive::Bool = false
                                )::Union{Nothing, Tuple{Vector{Int}, Vector{Int}}} where {A}
    m, n = size(M)
    (m + n < 8 || m < 2 || n < 2) && return nothing
    N = m + n
    w = _SccWork(N)
    skip = falses(N)
    row_sup = [findall(!iszero, @view M[r, :]) for r in 1:m]
    col_sup = [findall(!iszero, @view M[:, c]) for c in 1:n]
    no_extra = Int[]
    hi = N - 6                               # nodes: N - 4; at least 2 stay out
    rows_of(set) = [v for v in set if v <= m]
    cols_of(set) = [v - m for v in set if v > m]

    for (i1, j1) in _two_separation_pivots(M)
        p = M[i1, j1]

        # rank B = rank C = 1: (i2, j2) a nonzero of C, i2 on side 2, j2 on side 1.
        seeds = if exhaustive
            [(i, j) for j in 1:n for i in 1:m if i != i1 && j != j1 && M[i, j] != 0]
        else
            in_B = (r, c) -> M[r, j1] != 0 && M[i1, c] != 0 && p * M[r, c] == M[r, j1] * M[i1, c]
            forest, rcomp, ccomp = _rule_forest(m, n, i1, j1, (r, c) -> M[r, c] != 0 && !in_B(r, c))
            for c in 1:n, r in 1:m
                (r == i1 || c == j1 || M[r, c] == 0 || rcomp[r] == ccomp[c]) && continue
                in_B(r, c) && push!(forest, (r, c))
            end
            forest
        end
        for (i2, j2) in seeds
            q = M[i2, j2]
            fill!(skip, false)
            skip[i1] = skip[i2] = skip[m + j1] = skip[m + j2] = true
            rowedge = (r, c) -> @inbounds p * M[r, c] != M[r, j1] * M[i1, c]
            coledge = (c, r) -> @inbounds q * M[r, c] != M[r, j2] * M[i2, c]
            _rule_sccs!(w, m, n, skip, rowedge, coledge,
                        row_sup, row_sup[i1], col_sup, col_sup[j2])
            side = set -> (sort!([i1; rows_of(set)]), sort!([j2; cols_of(set)]))
            set = _closed_set(w, m, n, skip, 2, hi, rowedge, coledge, set -> accept(side(set)...))
            set !== nothing && return side(set)
        end

        # rank B = 2, C = 0: rows i1, i2 on side 1, columns j1, j2 on side 2.
        schur = (r, c) -> p * M[r, c] - M[r, j1] * M[i1, c]
        seeds = if exhaustive
            [(i, j) for j in 1:n for i in 1:m if i != i1 && j != j1 && schur(i, j) != 0]
        else
            forest, rcomp, ccomp = _rule_forest(m, n, i1, j1, (r, c) -> M[r, c] != 0 && schur(r, c) != 0)
            for c in 1:n, r in 1:m
                (r == i1 || c == j1 || M[r, c] != 0 || rcomp[r] == ccomp[c]) && continue
                schur(r, c) != 0 && push!(forest, (r, c))
            end
            forest
        end
        for (i2, j2) in seeds
            x11, x12, x21, x22 = p, M[i1, j2], M[i2, j1], M[i2, j2]
            d = x11 * x22 - x12 * x21
            fill!(skip, false)
            skip[i1] = skip[i2] = skip[m + j1] = skip[m + j2] = true
            # 3×3 minor on rows i1, i2, r and columns j1, j2, c, expanded
            # along the last row and column.
            rowedge = (r, c) -> @inbounds M[r, c] * d !=
                M[r, j1] * (x22 * M[i1, c] - x12 * M[i2, c]) +
                M[r, j2] * (x11 * M[i2, c] - x21 * M[i1, c])
            coledge = (c, r) -> @inbounds M[r, c] != 0
            _rule_sccs!(w, m, n, skip, rowedge, coledge,
                        row_sup, vcat(row_sup[i1], row_sup[i2]), col_sup, no_extra)
            side = set -> (sort!([i1; i2; rows_of(set)]), sort!(cols_of(set)))
            set = _closed_set(w, m, n, skip, 2, hi, rowedge, coledge, set -> accept(side(set)...))
            set !== nothing && return side(set)
        end
    end
    return nothing
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
        # Am and Bm need at least one row and one column each.
        (size(A, 1) < 2 || size(A, 2) < 3) &&
            error("Matrix A does not have the required form for a 3-sum.")
        (size(B, 1) < 2 || size(B, 2) < 3) &&
            error("Matrix B does not have the required form for a 3-sum.")
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

# Convenience methods: accept any integer matrices (Int8, adjoints, views, …).
for f in (:one_sum, :two_sum, :three_sum)
    @eval $f(A::AbstractMatrix{<:Integer}, B::AbstractMatrix{<:Integer}) =
        $f(Matrix{Int}(A), Matrix{Int}(B))
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
every square submatrix of `M` has determinant in {-1, 0, 1}.

After reduction and splitting into connected blocks, each block is tested for
being a network matrix, the transpose of one, or one of the special matrices
[`F_1`](@ref), [`F_2`](@ref) (Schrijver, *Theory of Linear and Integer
Programming*, Theorems 20.1 and 20.3). Blocks that are none of these are
split along a 2-separation (2-sum) or, if there is none, a 3-separation
(3-sum), both found in polynomial time, and the pieces are tested in the same
way; a block with neither is not TU. Blocks whose smaller dimension is at
most 8 are decided by an exact branch-and-prune Ghouila-Houri test instead.
`cmr_is_totally_unimodular(M; algorithm = :decomposition)` runs an
independent, exhaustive separation search on blocks up to 12×12.

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
    _is_tu_recursive(M, 0, Set{Matrix{Int}}(), true)
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
    # Out-of-range entries: still go through the Matrix{Int} method so the
    # keyword arguments are validated; any out-of-range stand-in gives false.
    cmr_is_totally_unimodular(N === nothing ? fill(2, 1, 1) : N; kwargs...)
end

function _is_tu_recursive(M::Matrix{Int}, depth::Int, seen::Set{Matrix{Int}}, fast::Bool)::Bool
    ok, M = _reduce(M)
    ok || return false
    (size(M, 1) == 0 || size(M, 2) == 0) && return true
    # `depth` counts pivot steps (Cases 5/6) only: every sum case recurses on
    # strictly smaller matrices, so only pivots can run away. Runaway
    # recursion says nothing about M; decide it exactly instead.
    depth > 100 && return _tu_partition(M)

    # 1-sum: split into connected components and test each independently.
    # O(m·n) bipartite BFS — far cheaper than any subsequent step.
    let comps = _bipartite_components(M)
        if comps !== nothing
            return all(((rows, cols),) -> _is_tu_recursive(M[rows, cols], depth, seen, fast), comps)
        end
    end

    # Cycle detection: `seen` holds the matrices on the *current* recursion
    # path only. Membership means the pivot cases (5/6) have cycled back to an
    # ancestor, so this branch cannot make progress. That says nothing about
    # whether M is TU (TU matrices do cycle, e.g. degenerate 3-sum retry ↔
    # rank-2 pivot), so decide M exactly with the Ghouila-Houri test instead.
    # Each matrix is removed once its subtree is done — identical matrices in
    # sibling branches (e.g. duplicate blocks of a 1-sum) are legitimate.
    M in seen && return _tu_partition(M)
    push!(seen, M)
    result = _is_tu_irreducible(M, depth, seen, fast)
    delete!(seen, M)
    return result
end

# Blocks whose smaller dimension is at most this are decided by the exact
# Ghouila-Houri test instead of being searched for a 3-separation. Measured
# on TU blocks with no 2-separation, the search route is level at 8 and
# ahead beyond (4 ms against 6 ms at 15×13, 0.7 ms against 0.6 s at 21×19).
# A Ref so that tests can set it to 0 and compare the search route with the
# exact test on small matrices.
const _PARTITION_MAX_DIM = Ref(8)

# TU test for a matrix that is already reduced, connected, and not on the
# current recursion path.
function _is_tu_irreducible(M::Matrix{Int}, depth::Int, seen::Set{Matrix{Int}}, fast::Bool)::Bool
    _is_network_matrix(M) && return true
    _is_network_matrix(Matrix{Int}(M')) && return true
    _is_special_matrix(M) && return true

    # 2-sum splitting: a polynomial search for a 2-separation, so that the
    # exponential test below only ever sees blocks that cannot be split this
    # way — cost is then exponential in the largest such block rather than
    # in the whole matrix. With `fast = false`, blocks up to 12×12 skip this
    # and take the exhaustive Seymour decomposition search instead
    # (cmr_is_totally_unimodular's :decomposition).
    m, n = size(M)
    legacy = !fast && m <= 12 && n <= 12
    if !legacy
        sep = _find_two_separation(M)
        if sep !== nothing
            R1, C1 = sep
            R2 = setdiff(1:m, R1)
            C2 = setdiff(1:n, C1)
            return _apply_decomposition(M, M[R1, C1], M[R1, C2], M[R2, C1], M[R2, C2],
                                        1, 0, depth, seen, fast)
        end
    end

    # Quick non-TU detector: Eulerian check at k ≤ 3 catches most violations
    # (e.g. any 2×2 or 3×3 bad submatrix) before the exponential searches
    # run. It costs O(m³n³) in the worst case, so it runs after the 2-sum
    # split: on a splittable matrix only the pieces pay for it.
    _tu_eulerian(M, 3) || return false

    if !legacy
        # Small blocks: the exact branch-and-prune Ghouila-Houri test is
        # exponential in the smaller dimension but faster than searching
        # for a separation at these sizes.
        min(m, n) <= _PARTITION_MAX_DIM[] && return _tu_partition(M)

        # M is reduced, connected, has no 2-separation and is neither a
        # network matrix, the transpose of one, nor F_1/F_2. By Seymour's
        # theorem it is then TU only if it has a 3-separation, and
        # _find_three_separation is exhaustive.
        sep = _find_three_separation(M)
        sep === nothing && return false
        R1, C1 = sep
        R2 = setdiff(1:m, R1)
        C2 = setdiff(1:n, C1)
        A, B, C, D = M[R1, C1], M[R1, C2], M[R2, C1], M[R2, C2]
        return _apply_decomposition(M, A, B, C, D, _rank_int(B), _rank_int(C),
                                    depth, seen, fast)
    end

    # Exhaustive search, ≤12×12 only: no separation means not TU.
    found, (A, B, C, D) = _decompose(M)
    found || return false

    rB = _rank_int(B)
    rC = _rank_int(C)
    return _apply_decomposition(M, A, B, C, D, rB, rC, depth, seen, fast)
end

# Continue Cases 5/6 after the pivot: P is the pivoted matrix and (R1, C1),
# (R2, C2) the split carried over from before the pivot, which has
# rank(B) = rank(C) = 1 there.
function _apply_pivoted(P::Matrix{Int}, R1, C1, R2, C2,
                        depth::Int, seen::Set{Matrix{Int}}, fast::Bool)::Bool
    A, B, C, D = P[R1, C1], P[R1, C2], P[R2, C1], P[R2, C2]
    rB, rC = _rank_int(B), _rank_int(C)
    (rB == 1 && rC == 1) ||
        error("split of the pivoted matrix has rank(B) = $rB, rank(C) = $rC, expected 1 and 1")
    _apply_decomposition(P, A, B, C, D, 1, 1, depth + 1, seen, fast)
end

# Dispatch on the rank case of a decomposition M = [A B; C D] of a reduced,
# connected matrix M.
function _apply_decomposition(M::Matrix{Int},
                               A::Matrix{Int}, B::Matrix{Int},
                               C::Matrix{Int}, D::Matrix{Int},
                               rB::Int, rC::Int,
                               depth::Int, seen::Set{Matrix{Int}}, fast::Bool)::Bool
    if rB == 0 && rC == 0
        return _is_tu_recursive(A, depth, seen, fast) && _is_tu_recursive(D, depth, seen, fast)

    elseif rB == 1 && rC == 0
        f, g = _extract_rank1(B)
        return _is_tu_recursive([A f], depth, seen, fast) &&
               _is_tu_recursive([g; D], depth, seen, fast)

    elseif rB == 0 && rC == 1
        f, g = _extract_rank1(C)
        return _is_tu_recursive([A; g], depth, seen, fast) &&
               _is_tu_recursive([f D], depth, seen, fast)

    elseif rB == 1 && rC == 1
        # Case 4 needs M to be reduced, connected and free of 2-separations,
        # which holds here: the default route runs the 2-separation search
        # first and the exhaustive ≤12×12 search prefers splits of lower
        # rank, and a pivot (Cases 5/6) preserves all three. It does not need
        # A or D to be reduced themselves. An earlier version rejected splits
        # whose A or D had a trivial row or column or a duplicate pair, and
        # searched for another split; on larger matrices there is usually
        # none.
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
        # No R–K path means ε is undetermined, not that M is non-TU.
        (ok1 && ok2) || return _tu_partition(M)
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
        # First row of mat2: 1 over the B_cols block (D1/D3), 0 over the rest —
        # block widths must match the rows below (1, 1, nBK, nnotBK).
        mat2 = [ε₁                   0                     ones(Int,1,nBK)       zeros(Int,1,nnotBK)
                ones(Int,nCR,1)      ones(Int,nCR,1)       D1                    D2
                zeros(Int,nnotCR,1)  zeros(Int,nnotCR,1)   D3                    D4   ]
        return _is_tu_recursive(mat1, depth, seen, fast) &&
               _is_tu_recursive(mat2, depth, seen, fast)

    elseif rB == 2 && rC == 0
        # Case 5: pivot on a nonzero of B. A pivot exchanges the roles of its
        # row and its column, so the same split of the matroid's elements —
        # the pivot row now counted with side 2, the pivot column with side
        # 1 — is a split of the pivoted matrix, and there rank(B) =
        # rank(C) = 1: B becomes its own Schur complement, and C a multiple
        # of the pivot column of D times the pivot row of A. Case 4 is
        # applied to that split directly. Searching the pivoted matrix
        # afresh instead can return another rank-2 split and pivot straight
        # back.
        pivot_pos = findfirst(!iszero, B)
        pivot_pos === nothing && error("B has rank 2 but no nonzero entries")
        pi, pj = pivot_pos[1], pivot_pos[2]
        rA, cA = size(A)
        rD = size(D, 1)
        row_order = [pi; [i for i in 1:rA if i != pi]; collect(rA+1:rA+rD)]
        col_order = [cA+pj; collect(1:cA); [cA+j for j in 1:size(B,2) if j != pj]]
        M_full = [A B; zeros(Int,rD,cA) D]
        P = pivot(M_full[row_order, col_order], 1)
        all(x -> -1 <= x <= 1, P) || return false
        R1 = 2:rA;            C1 = 1:cA+1
        R2 = [1; rA+1:rA+rD]; C2 = cA+2:size(P, 2)
        return _apply_pivoted(P, R1, C1, R2, C2, depth, seen, fast)

    elseif rB == 0 && rC == 2
        # Case 6: as Case 5, pivoting on a nonzero of C. The pivot row now
        # counts with side 1 and the pivot column with side 2.
        pivot_pos = findfirst(!iszero, C)
        pivot_pos === nothing && error("C has rank 2 but no nonzero entries")
        pi, pj = pivot_pos[1], pivot_pos[2]
        rA, cA = size(A)
        rC_size = size(C, 1)
        cD = size(D, 2)
        row_order = [rA+pi; collect(1:rA); [rA+i for i in 1:rC_size if i != pi]]
        col_order = [pj; [j for j in 1:cA if j != pj]; collect(cA+1:cA+cD)]
        M_full = [A zeros(Int,rA,cD); C D]
        P = pivot(M_full[row_order, col_order], 1)
        all(x -> -1 <= x <= 1, P) || return false
        R1 = 1:rA+1;                 C1 = 2:cA
        R2 = rA+2:size(P, 1);        C2 = [1; cA+1:cA+cD]
        return _apply_pivoted(P, R1, C1, R2, C2, depth, seen, fast)

    else
        error("Unexpected rank(B) + rank(C) = $(rB + rC)")
    end
end


# ──────────────────────────────────────────────────────────────────────────────
# Partition algorithm (Ghouila-Houri criterion)
# Originally a port of tuPartition / tuPartitionSubset / tuPartitionSearch from
# cmr/tu.c. The subset enumeration and the per-subset sign search remain
# separate phases (see comments in _tu_partition for why they must); the sign
# search is branch-and-prune, which is orders of magnitude faster than the
# plain 3^r enumeration on typical inputs.
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
            # k rows chosen. Usable columns = those with even, nonzero count
            # of nonzeros. A column that is zero on the chosen rows can be
            # skipped: every minimal non-TU submatrix is Eulerian with sum
            # ≡ 2 (mod 4) and is nonsingular, so it has no zero column — the
            # criterion stays exact, and sparse matrices lose most candidates.
            n_use = 0
            for j in 1:c
                if col_nz[j] > 0 && col_nz[j] % 2 == 0; n_use += 1; use_cols[n_use] = j; end
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

**`:decomposition`** runs the Seymour decomposition of Theorem 20.3, with its
exhaustive separation search, on blocks up to 12×12 (where
[`is_totally_unimodular`](@ref) uses its polynomial separation searches);
larger blocks are handled as in `is_totally_unimodular`.

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
    algorithm in (:decomposition, :eulerian, :partition) ||
        throw(ArgumentError("Unknown algorithm $(repr(algorithm)). " *
                            "Use :decomposition, :eulerian, or :partition."))
    all(m -> m in (-1, 0, 1), M) || return false
    if algorithm === :decomposition
        return _is_tu_recursive(M, 0, Set{Matrix{Int}}(), false)
    elseif algorithm === :eulerian
        return _tu_eulerian(M)
    else
        return _tu_partition(M)
    end
end


end # module TotalUnimodularity
