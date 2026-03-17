module TotalUnimodularity

using LinearAlgebra
using Combinatorics

const F_1 = [1 -1 0 0 -1; -1 1 -1 0 0; 0 -1 1 -1 0; 0 0 -1 1 -1; -1 0 0 -1 1]
const F_2 = [1 1 1 1 1; 1 1 1 0 0; 1 0 1 1 0; 1 0 0 1 1; 1 1 0 0 1]


function pivot(M::Matrix{Int}; row::Int = 1, col::Int = 1)
    r, c = size(M)
    B = Matrix{Int}(I, r, r)
    @views B[:, 1] = M[:, 1]
    round.(Int, B \ [-(1:r .== 1) M[:, 2:end]])
end

function pivot(M::Matrix{Int}, k::Int)
    r, c = size(M)
    @views E, B, C, D = M[1:k, 1:k], M[k+1:end, 1:k], M[1:k, k+1:end], M[k+1:end, k+1:end]
    Einv = round.(Int, inv(E))
    [-Einv Einv*C; B*Einv D - B*Einv*C]
end

check_size(A::Matrix{Int}) = sum(size(A)) < 4 ? error("The number of rows plus columns of each matrix must be at least four.") : size(A)

function one_sum(A::Matrix{Int}, B::Matrix{Int})
    (rA, cA), (rB, cB) = check_size.((A, B))
    C = zeros(Int, rA + rB, cA + cB)
    C[1:rA, 1:cA] = A
    C[(rA+1):(rA+rB), (cA+1):(cA+cB)] = B
    return C
end

function two_sum(A::Matrix{Int}, B::Matrix{Int})
    check_size.((A, B))
    @views Am, a = A[:, 1:end-1], A[:, end]
    @views Bm, b = B[2:end, :], B[1, :]
    rA, cA = size(Am)
    rB, cB = size(Bm)
    C = zeros(Int, rA + rB, cA + cB)
    @views C[1:rA, 1:cA] = Am
    @views C[(rA+1):(rA+rB), (cA+1):(cA+cB)] = Bm
    @views C[1:rA, (cA+1):(cA+cB)] = (a*b')
    return C
end

function three_sum(A::Matrix{Int}, B::Matrix{Int})
    check_size.((A, B))
    @views ((A[1:end-1, end-1] != A[1:end-1, end]) ||
            (A[end, end-1] != 0) || (A[end, end] != 1)) &&
            error("The matrix A does not have the required form.")
    @views ((B[2:end, 1] != B[2:end, 2]) || (B[1, 1] != 1) || (B[1, 2] != 0)) &&
        error("The matrix B does not have the required form.")
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


is_standard_basis_vector(v) = count(x -> x != 0, v) == 1
is_zero_vector(v) = all(v[i] == 0 for i in 1:length(v))
is_standard_basis_vector_or_zero(v) = count(x -> x != 0, v) <= 1

function drop_standard_basis_vectors_and_zeroes(M)
    M = M[:, [!is_standard_basis_vector_or_zero(M[:, i]) for i in 1:size(M, 2)]]
    M[[!is_standard_basis_vector_or_zero(M[i, :]) for i in 1:size(M, 1)], :]
end

function recursive_drop_standard_basis_vectors_and_zeroes(M::Matrix{Int})
    N = drop_standard_basis_vectors_and_zeroes(M)
    while N != M
        M = N
        N = drop_standard_basis_vectors_and_zeroes(M)
    end
    N
end
    

function is_totally_unimodular(M::Matrix{Int}, verbose = false)
    !all(m in (-1, 0, 1) for m in M) && return false
    # for i=1:size(M, 1)
    #     if count(x->x != 0, M[i, :]) <= 1
    #         M = M[1:end .!=i, :]
    #     end
    # end
    # @show M
    # for i=1:size(M, 2)
    #     if count(x->x != 0, M[:, i]) <= 1
    #         M = M[:, 1:end .!=i]
    #     end
    # end
    # @show M
    r, c = size(M)
    n = min(r,c)
    
    # for s = 1:n
    #     for i in combinations(1:r, s), j in combinations(1:c, s)
    #         @views !in(det(M[i, j]), (-1, 0, 1)) && return false
    #     end
    # end
    return true
end


function naive_is_totally_unimodular(M::Matrix{Int})
    r, c = size(M)
    n = min(r,c)
    
    for s = 1:n
        for i in combinations(1:r, s), j in combinations(1:c, s)
            @views !in(det(M[i, j]), (-1, 0, 1)) && return false
        end
    end
    return true
end

end
