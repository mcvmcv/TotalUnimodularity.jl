module TotalUnimodularity

using LinearAlgebra
using Combinatorics

# Public API
export naive_is_totally_unimodular
export is_totally_unimodular
export one_sum, two_sum, three_sum
export pivot
export F_1, F_2

# -----------------------------------------------------------
# Special matrices (Seymour's theorem)
# -----------------------------------------------------------
"""
F_1

The first special totally unimodular matrix in Seymour's decomposition theorem.
This 5x5 matrix is TU but cannot be decomposed via 1-, 2- 3-sums.
"""
const F_1 = [ 1 -1  0  0 -1
             -1  1 -1  0  0
              0 -1  1 -1  0
              0  0 -1  1 -1
             -1  0  0 -1  1]

"""
F_2

The second special totally unimodular matrix in Seymour's decomposition theorem.
This 5x5 matrix is TU but cannot be decomposed via 1-, 2- 3-sums.
"""
const F_2 = [1 1 1 1 1
             1 1 1 0 0
             1 0 1 1 0
             1 0 0 1 1
             1 1 0 0 1]

# -----------------------------------------------------------
# Internal helpers
# -----------------------------------------------------------

"""
Check that a matrix has at least 2 rows and 2 columns (r+c >= 4).
Returns (r, c) if valid, otherwise throws an error.
"""
function _check_size(A::Matrix{Int})
    r, c = size(A)
    r + c < 4 && error("The number of rows plus columns of each matrix must be at least four.")
    return r, c
end

"""
Return true if vector v is a standard basis vector (exactly one nonzero entry).
"""
_is_standard_basis_vector(v::AbstractVector) = count(!iszero, v) == 1

"""
Return true if vector v is the zero vector.
"""
_is_zero_vector(v::AbstractVector) = all(iszero, v)

"""
Return true if vector v is a standard basis vector of the zero vector.
"""
_is_trivial_vector(v::AbstractVector) = count(!iszero, v) <= 1

"""
Drop all columns and rows that are standard basis vectors or zero vectors.
"""
function _drop_trivial_vectors(M::Matrix{Int})
    col_mask = [!_is_trivial_vector(@view M[:, j]) for j in 1:size(M, 2)]
    M = M[:, col_mask]
    row_mask = [!_is_trivial_vector(@view M[i, :]) for i in 1:size(M, 1)]
    return M[row_mask, :]
end

"""
Repeatedly drop trivial rows and columns until the matrix stabilises.
"""
function _reduce_trivial_vectors(M::Matrix{Int})
    while true
        N = _drop_trivial_vectors(M)
        N == M && return M
        M = N
    end
end


# -----------------------------------------------------------
# Pivot operation
# -----------------------------------------------------------

"""
pivot(M, k)

Perform the pivot operation on a matrix `M` with respect to the leading kxk
submatrix E, assumed to be invertible with determinant +-1 (as holds for TU
matrices).

Given the partition M = [E C; B D], returns [-E^-1, E^-1C; BE^-1, D-BE^-1C].

This operation preserves total unimodularity and is used in Seymour
decomposition.

# Arguments
- `M`: An integer matrix
- `k`: Size of the leading square submatrix to pivot on

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function pivot(M::Matrix{Int}; k::Int)
    r, c = size(M)
    @views E = M[1:k,     1:k    ]
    @views B = M[k+1:end, 1:k    ]
    @views C = M[1:k,     k+1:end]
    @views D = M[k+1:end, k+1:end]
    Einv = Matrix{Int}(round.(inv(Matrix{Rational{Int}}(E))))
    [-Einv      Einv*C
     B*Einv     D-B*Einv*C]
    B = Matrix{Int}(I, r, r)
end

# -----------------------------------------------------------
# Seymour decomposition operations
# -----------------------------------------------------------

"""
one_sum(A, B)

Compute the 1-sum of matrices `A` and `B`.

The 1-sum is the block diagonal matrix [A 0; 0 B]. If A and B are both TU,
then so is their 1-sum.

# Arguments
- `A`, `B`: Integer mattrices, each with r+c >= 4.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function one_sum(A::Matrix{Int}, B::Matrix{Int})
    rA, cA = _check_size(A)
    rB, cB = _check_size(B)
    C = zeros(Int, rA + rB, cA + cB)
    C[1:rA, 1:cA] = A
    C[(rA+1):(rA+rB), (cA+1):(cA+cB)] = B
    return C
end

"""
two_sum(A, B)

Compute the 2-sum of matrices `A` and `B`.

A must have a distinguished last column `a` and B a distinguished first
row `b^T`. The 2-sum replaces the last column of A and first row of B
with the outer product `axb`, combining the matrices.

If A and B are TU, then so is their 2-sum.

# Arguments
- `A`, `B`: Integer matrices, each with r+c >= 4.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function two_sum(A::Matrix{Int}, B::Matrix{Int})
    _check_size(A)
    _check_size(B)
    @views Am, a = A[:, 1:end-1], A[:, end]
    @views b, Bm = B[1, :],       B[2:end, :]
    rA, cA = size(Am)
    rB, cB = size(Bm)
    C = zeros(Int, rA + rB, cA + cB)
    @views C[1:rA, 1:cA] = Am
    @views C[(rA+1):(rA+rB), (cA+1):(cA+cB)] = Bm
    @views C[1:rA, (cA+1):(cA+cB)] = a*b'
    return C
end

"""
three_sum(A, B)

Compute the 3-sum of matrices `A` and `B`.

A must have the form [Am a a; c 0 1] and B must have the form
[1 0 b; d d Bm], where a, b, c, d are vectors. The 3-sum combines
these matrices by eliminating the shared structure.

If A and B are TU, then so is their 3-sum.

# Arguments
- `A`, `B`: Integer matrices in the required form; throws an error otherwise.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function three_sum(A::Matrix{Int}, B::Matrix{Int})
    _check_size(A)
    _check_size(B)
    @views begin
        (A[1:end-1, end-1] != A[1:end-1, end] ||
         A[end, end-1] != 0) || A[end, end] != 1) &&
            error("The matrix A does not have the required form for a 3-sum.")
        (B[2:end, 1] != B[2:end, 2] ||
        B[1, 1] != 1 || B[1, 2] != 0) &&
        error("The matrix B does not have the required form for a 3-sum.")
    end
    @views Am, a, c = A[1:end-1, 1:end-2], A[1:end-1, end], A[end, 1:end-2]
    @views Bm, b, d = B[2:end, 3:end], B[1, 3:end], B[2:end, 1]
    rA, cA = size(Am)
    rB, cB = size(Bm)
    C = zeros(Int, rA + rB, cA + cB)
    @views C[1:rA, 1:cA] = Am
    @views C[1:rA, (cA+1):(cA+cB)] = (a*b')
    @views C[(rA+1):(rA+rB), 1:cA] = (d*c')
    @views C[(rA+1):(rA+rB), (cA+1):(cA+cB)] = Bm
    return C
end    

# -----------------------------------------------------------
# Total unimodularity tests
# -----------------------------------------------------------

"""
naive_is_totally_unimodular(M)

Test whether integer matrix `M` is totally unimodular by checking that
every square submatrix has determinant in {-1, 0, 1}.

This is correct but has exponential time complexity. Use for testing and
validation only; see [`is_totally_unimodular`](@ref) for the linear-time
implementation.

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function naive_is_totally_unimodular(M::Matrix{Int})
    r, c = size(M)
    n = min(r,c)
    
    for s = 1:n
        for rows in combinations(1:r, s), cols in combinations(1:c, s)
            @views det(M[rows, cols]) in (-1, 0, 1) && return false
        end
    end
    return true
end

"""
is_totally_unimodular(M)

Test whether integer matrix `M` is totally unimodular using a linear-time
algorithm based on Seymour's decomposition theorem.

A matrix is totally unimodular if and only if it can be constructed from
network matrices, their transposes, [`F_1`](@ref), and [`F_2`](@ref) using
[`one_sum`](@ref), [`two_sum`](@ref), and [`three_sum`](@ref).

# Reference
Schrijver, *Theory of Linear and Integer Programming*, Chapter 20.
"""
function is_totally_unimodular(M::Matrix{Int})
    !all(m in (-1, 0, 1) for m in M) && return false
    # TODO: Implement linear-time TU test via Seymour decomposition.
    error("Not yet implemented")
end

end


