module TotalUnimodularity

using LinearAlgebra
using Combinatorics

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

Test whether integer matrix `M` is totally unimodular using a linear-time
algorithm based on Seymour's decomposition theorem.

A matrix is totally unimodular if and only if it can be constructed from
network matrices, their transposes, [`F_1`](@ref), and [`F_2`](@ref) via
[`one_sum`](@ref), [`two_sum`](@ref), and [`three_sum`](@ref).

# Arguments
- `M::Matrix{Int}`: An integer matrix whose entries must be in {-1, 0, 1};
  returns `false` immediately otherwise.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Theorem 20.3.
"""
function is_totally_unimodular(M::Matrix{Int})
    all(m -> m in (-1, 0, 1), M) || return false
    # TODO: implement linear-time TU test via Seymour decomposition
    error("Not yet implemented")
end

end # module TotalUnimodularity
