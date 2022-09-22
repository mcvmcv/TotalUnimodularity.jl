module TotalUnimodularity

using LinearAlgebra
using Combinatorics

function is_totally_unimodular(M::Matrix{Int}, verbose = false)
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
