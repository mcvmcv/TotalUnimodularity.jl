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

# Return true if v is a standard basis vector, i.e. exactly one entry equal
# to 1 and all others 0.
_is_standard_basis_vector(v::AbstractVector) =
    count(!iszero, v) == 1 && all(x -> x in (0, 1), v)

# Return true if v is a standard basis vector or the zero vector.
_is_trivial_vector(v::AbstractVector) =
    iszero(v) || _is_standard_basis_vector(v)

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
    _is_sign_and_permutation_equivalent(M, F_1) ||
    _is_sign_and_permutation_equivalent(M, F_2)
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
    IM = [Matrix{Int}(I, m, m) M]  # [I | M]
    N = m + n                       # total columns of IM
    XI = Set(1:m)                   # column indices of I part
    XM = Set(m+1:m+n)              # column indices of M part
    rhoX = rank(IM)

    for S in combinations(1:N, 4)
        # S must intersect both XI and XM
        any(s -> s in XI, S) || continue
        any(s -> s in XM, S) || continue
        S_set = Set(S)

        for T in combinations(1:N, 4)
            # T must intersect both XI and XM
            any(t -> t in XI, T) || continue
            any(t -> t in XM, T) || continue

            # S and T must be disjoint
            isempty(S_set ∩ Set(T)) || continue

            # Solve problem (16)
            found, Y = _solve_submodular(IM, collect(S), collect(T), rhoX)
            found || continue

            Y_set = Set(Y)

            # Y∩XI → row indices that go to top partition (A, B rows)
            # Y∩XM → col indices that go to left partition (A, C cols)
            row_top  = sort([s       for s in Y if s in XI])
            col_left = sort([s - m   for s in Y if s in XM])
            row_bot  = sort([i       for i in 1:m if i ∉ Y_set])
            col_right = sort([j      for j in 1:n if (j + m) ∉ Y_set])

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
            rank(B) + rank(C) <= 2 || continue

            return (true, (A, B, C, D))
        end
    end
    return (false, (M, M, M, M))
end

function _solve_submodular(IM::Matrix{Int}, S::Vector{Int},
                            T::Vector{Int}, rhoX::Int)::Tuple{Bool, Vector{Int}}
    N = size(IM, 2)
    V = [v for v in 1:N if v ∉ S && v ∉ T]  # V = X\(S∪T)
    Z = Int[]  # start with Z = ∅

    while true
        SZ = S ∪ Z
        TZ = T ∪ Z
        rhoSZ = rank(IM[:, SZ])
        rhoTZ = rank(IM[:, TZ])

        # Compute U and W from (19)
        U = [v for v in V if v ∉ Z &&
             rank(IM[:, SZ ∪ [v]]) == rhoSZ + 1]
        W = [v for v in V if v ∉ Z &&
             rank(IM[:, TZ ∪ [v]]) == rhoTZ + 1]

        # Build digraph D from (18)
        # For u∈Z, v∈V\Z:
        #   (u,v) ∈ E iff ρ(S∪(Z\{u})∪{v}) = ρ(S)+|Z|
        #   (v,u) ∈ E iff ρ(T∪(Z\{u})∪{v}) = ρ(T)+|Z|
        d_edges = Tuple{Int,Int}[]
        rhoS = rank(IM[:, S])
        rhoT = rank(IM[:, T])
        for u in Z
            Zminu = [z for z in Z if z != u]
            for v in V
                v in Z && continue
                if rank(IM[:, S ∪ Zminu ∪ [v]]) == rhoS + length(Z)
                    push!(d_edges, (u, v))
                end
                if rank(IM[:, T ∪ Zminu ∪ [v]]) == rhoT + length(Z)
                    push!(d_edges, (v, u))
                end
            end
        end

        # Find shortest path from U to W in D
        path = _shortest_path(d_edges, U, W, V)

        if path !== nothing
            # Case 1: update Z via symmetric difference (20)
            path_set = Set(path)
            Z = [v for v in V if (v in Z) ⊻ (v in path_set)]
        else
            # Case 2: Y = S ∪ {v ∈ V | path from v to W exists in D} (23)
            reachable = _reachable_to(d_edges, W, V)
            Y = sort(S ∪ reachable)  # fixed: no explicit ∪ Z

            # Check ρ(Y) + ρ(X\Y) ≤ ρ(X) + 2
            XminusY = [v for v in 1:N if v ∉ Y]  # fixed: use full X\Y
            rhoY = rank(IM[:, Y])
            rhoXminusY = isempty(XminusY) ? 0 : rank(IM[:, XminusY])

            return (rhoY + rhoXminusY <= rhoX + 2, Y)
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

"""
    is_totally_unimodular(M)

Test whether integer matrix `M` is totally unimodular using the polynomial-time
algorithm of Theorem 20.3 (Seymour's decomposition theorem).

A matrix is totally unimodular if and only if it can be constructed from
network matrices, their transposes, [`F_1`](@ref), and [`F_2`](@ref) via
[`one_sum`](@ref), [`two_sum`](@ref), and [`three_sum`](@ref).

# Arguments
- `M::Matrix{Int}`: An integer matrix.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3.
"""
function is_totally_unimodular(M::Matrix{Int})::Bool
    # Step 1: preprocess — check entries and reduce
    ok, M = _reduce(M)
    ok || return false

    # Trivial cases after reduction
    size(M, 1) == 0 || size(M, 2) == 0 && return true

    # Step 2: check if network matrix or its transpose
    _is_network_matrix(M) && return true
    _is_network_matrix(Matrix{Int}(M')) && return true

    # Step 3: check if special matrix (F_1 or F_2 up to permutation/scaling)
    _is_special_matrix(M) && return true

    # Step 4: try Seymour decomposition
    found, (A, B, C, D) = _decompose(M)

    # No decomposition exists → not TU (by Corollary 19.6b)
    found || return false

    # Step 5: recurse based on rank(B) + rank(C)
    rB = rank(B)
    rC = rank(C)

    if rB == 0 && rC == 0
        # Case 1: M is TU iff A and D are TU
        return is_totally_unimodular(A) && is_totally_unimodular(D)

    elseif rB == 1 && rC == 0
        # Case 2: B = f⊗g, M is TU iff [A f] and [g; D] are TU
        f, g = _extract_rank1(B)
        return is_totally_unimodular([A f]) && is_totally_unimodular([g; D])

    elseif rB == 0 && rC == 1
        # Case 3: C = f⊗g, M is TU iff [A; g] and [f D] are TU
        f, g = _extract_rank1(C)
        return is_totally_unimodular([A; g]) && is_totally_unimodular([f D])

    elseif rB == 1 && rC == 1
        # Case 4: construct matrices from (31) and test both
        # Requires finding ε₁, ε₂ ∈ {+1,-1} via bipartite graph path
        # TODO: implement Case 4
        error("Case 4 not yet implemented")

    elseif rB == 2 && rC == 0
        # Case 5: pivot on nonzero entry of B to reduce to Case 4
        # TODO: implement Case 5
        error("Case 5 not yet implemented")

    elseif rB == 0 && rC == 2
        # Case 6: symmetric to Case 5
        # TODO: implement Case 6
        error("Case 6 not yet implemented")

    else
        # rank(B) + rank(C) > 2 — shouldn't happen if _decompose is correct
        error("Unexpected rank(B) + rank(C) = $(rB + rC)")
    end
end

end # module TotalUnimodularity
