using TotalUnimodularity
using Test
using LinearAlgebra
using Random

# Counts of fallbacks to the exact test at the start; see the last testset.
const fallbacks_before = map(x -> x[], TotalUnimodularity._FALLBACKS)

include("internals.jl")
include("test_cmr.jl")

@testset "Public API" begin

     @testset "Known TU matrices" begin
         @test naive_is_totally_unimodular(Matrix{Int}(I, 3, 3))
         @test naive_is_totally_unimodular(F_1)
         @test naive_is_totally_unimodular(F_2)
         @test naive_is_totally_unimodular([1 0 1; 0 1 1; -1 0 0])
         @test !naive_is_totally_unimodular([1 0 1; 1 1 0; 0 1 1])
         @test !naive_is_totally_unimodular([1 0 0 1; 1 0 1 0; 1 1 0 0; 1 1 1 1])
     end  

    @testset "is_totally_unimodular" begin
        @test is_totally_unimodular(Matrix{Int}(I, 3, 3))
        @test is_totally_unimodular(F_1)
        @test is_totally_unimodular(F_2)
        @test is_totally_unimodular(network_matrix)
        @test is_totally_unimodular(M3)
        @test is_totally_unimodular(one_sum(network_matrix, Matrix{Int}(I, 2, 2)))

        @test !is_totally_unimodular([1 1 0; 1 0 1; 0 1 1])
        @test !is_totally_unimodular([1 0 0 1; 1 0 1 0; 1 1 0 0; 1 1 1 1])
        @test !is_totally_unimodular([2 0; 0 1])  # entry outside {-1,0,1}

        # TU but not a network matrix (exercises special/decomposition path)
        @test is_totally_unimodular(non_network_tu)
        @test is_totally_unimodular(Matrix{Int}(network_matrix'))
        @test is_totally_unimodular(Matrix{Int}(M3'))
    end

    @testset "generic integer matrix inputs" begin
        M = [1 0 1 0; 0 1 1 0; 0 0 1 1]
        @test is_totally_unimodular(Int8.(M))
        @test is_totally_unimodular(M .== 1)            # BitMatrix
        @test is_totally_unimodular(transpose(M))       # lazy wrapper
        @test naive_is_totally_unimodular(Int32.(M))
        @test !is_totally_unimodular([Int128(2)^70 0; 0 1])  # out-of-range entry
    end

    @testset "argument validation" begin
        # Unknown algorithm is an error regardless of the entries.
        @test_throws ArgumentError cmr_is_totally_unimodular([1 0; 0 1]; algorithm=:bogus)
        @test_throws ArgumentError cmr_is_totally_unimodular([2 0; 0 1]; algorithm=:bogus)
        @test_throws ArgumentError cmr_is_totally_unimodular(Int8[2 0; 0 1]; algorithm=:bogus)
        @test !cmr_is_totally_unimodular(Int8[2 0; 0 1]; algorithm=:partition)
        # pivot: k out of range, or leading block not unimodular.
        @test_throws ArgumentError pivot([1 1; 1 1], 3)
        @test_throws ArgumentError pivot([1 1; 1 1], -1)
        @test_throws ArgumentError pivot([0 1; 1 1], 1)
        @test_throws ArgumentError pivot([1 1; 1 1], 2)
        @test pivot([1 1; 1 0], 1) == [-1 1; 1 -1]
        @test pivot([1 1; 1 0], 2) == -[0 1; 1 -1]
        # 3-sum summands too small to contain Am / Bm.
        @test_throws ErrorException three_sum([1 1; 0 1], [1 0 1; 1 1 0])
        @test_throws ErrorException three_sum([1 1 1; 1 0 1], [1 0; 1 1])
    end

    @testset "generic integer matrix inputs (sums and pivot)" begin
        A = [1 0 1; -1 1 0; 0 -1 -1]
        B = [1 1 0; 1 0 1; 1 -1 1]
        @test one_sum(Int8.(A), B) == one_sum(A, B)
        @test two_sum(A, Int8.(B)) == two_sum(A, B)
        @test two_sum(transpose(A), view(B, :, :)) == two_sum(Matrix{Int}(A'), B)
        A3 = [1 1 1; 1 0 1]
        B3 = [1 0 1; 1 1 0]
        @test three_sum(Int8.(A3), B3 .== 1) == three_sum(A3, B3)
        @test pivot(Int8.(A), 1) == pivot(A, 1)
        @test pivot(transpose(A), 1) == pivot(Matrix{Int}(A'), 1)
    end

    # The regressions below are bugs in the Seymour decomposition code, so they
    # use tu_both_routes (test_cmr.jl) to cover the :decomposition route too.
    # Regression: `seen` used to be a global visited-set, so identical blocks
    # in sibling branches (e.g. a 1-sum of a matrix with itself) were treated
    # as cycles and wrongly reported non-TU. Cycle detection is now path-based.
    @testset "duplicate blocks in sums" begin
        @test tu_both_routes(one_sum(K33, K33))
        @test tu_both_routes(one_sum(network_matrix, network_matrix))
        @test tu_both_routes(one_sum(one_sum(network_matrix, network_matrix),
                                            network_matrix))
        @test tu_both_routes(one_sum(F_1, F_1))
    end

    # Regression: _extract_rank1 assumed every nonzero column of a rank-1
    # block equals f, but B = f⊗g with g containing -1 entries has -f columns.
    # A 2-sum whose glue row has mixed signs used to give false negatives.
    # (All 32 column masks and 32 row masks were verified offline; a subset is
    # tested here to keep suite runtime down. Mask 2 is the original failure.)
    @testset "signed rank-1 glue blocks" begin
        for mask in (2, 5, 21, 31)   # ±1 column scalings of the right summand
            K33dx = copy(K33dual_twosum)
            for j in 1:5
                (mask >> (j - 1)) & 1 == 1 && (K33dx[:, j] .*= -1)
            end
            @test tu_both_routes(two_sum(K33, K33dx))
        end
        for mask in (0b00110, 0b10101)   # ±1 row scalings of the left summand
            K33x = copy(K33)
            for i in 1:5
                (mask >> (i - 1)) & 1 == 1 && (K33x[i, :] .*= -1)
            end
            @test tu_both_routes(two_sum(K33x, K33dual_twosum))
        end
    end

    # Regression: a pivot cycle (degenerate 3-sum retry ↔ rank-2 pivot
    # returning to an ancestor matrix) used to be reported as non-TU. These
    # TU matrices (2-sums of F_1/F_2/K33/network blocks with random pivots)
    # were found by structured fuzzing against the Ghouila-Houri test.
    @testset "pivot cycles on TU matrices" begin
        cyc1 = [-1 -1  0  0 -1  0  1  1
                 0 -1  1  0  0  1  0  0
                 0  0  1  0  0  1 -1  0
                 0  0  0  1 -1  0  0  1
                 1  0  0  0  1  1 -1 -1
                -1  0  0 -1  0  0  1  0
                -1  0  0  0 -1  0  0  0
                 0  0  0  1  0  0  0  1]
        cyc2 = [-1 -1  0  0  1 -1 -1  0
                 0 -1  1  0  1 -1 -1  0
                 0  0  1  0  1  0 -1  0
                 0  0  0  1  1  0  0  1
                 0  1  0 -1 -1  1  0 -1
                 0  1  0 -1 -1  0  0  0
                -1 -1  0  0  1 -1  0  0
                -1  0 -1  0  0  0  0  0]
        cyc3 = [-1 -1  0  0  0  0  0 -1
                -1  0  0  0  0 -1  0 -1
                 0  1  0  0  0 -1  0  0
                 0 -1 -1  0  1  0  0 -1
                -1  0  0  0  0 -1  0  0
                 0  0  0  1 -1  0 -1  0
                 0 -1 -1  1  0  0  0 -1
                 0  1  1 -1  0  0  1  0]
        for M in (cyc1, cyc2, cyc3)
            @test naive_is_totally_unimodular(M)
            @test tu_both_routes(M)
            @test tu_both_routes(Matrix{Int}(M'))
        end
    end

    # Regression: the second Case 4 (3-sum) matrix was assembled with its first
    # row's blocks in the wrong widths (1, nBK, nnotBK, 1 instead of
    # 1, 1, nBK, nnotBK), so the 1s sat over the wrong columns of D whenever
    # nBK ≠ nnotBK and a TU matrix could yield a non-TU summand. Here the
    # non-degenerate 3-sum partition has nBK = 1, nnotBK = 3.
    @testset "3-sum with unequal B-column split" begin
        M8 = [ 1  0 -1 -1  0  0 -1  0
               1 -1  0  0  0  0 -1  0
               0 -1  0  1  0  0  0  0
              -1  1  1  0  0  0  1  0
               0  0  1  1  0 -1  1  0
               0  0  0  0 -1  0  1 -1
               0  0 -1 -1  0  0 -1  1
               0  0  0  0 -1  1  0 -1]
        @test naive_is_totally_unimodular(M8)
        @test tu_both_routes(M8)
        @test tu_both_routes(Matrix{Int}(M8'))
        # The 12×11 fuzz input that reduces to M8 via a 2-sum.
        M12 = [ 0  0  0 -1  0  0  0  0  0 -1  0
                0 -1  1  0  0  0  0  1  1  0  0
                0  0  0  1  0  0 -1  0  0  1  0
                0  0  0  0 -1 -1 -1  1  1  1  0
               -1  0  1  0  0  0  0  0  0  0 -1
                0  0  0  0  0 -1 -1  0  1  1  0
                0  0  1  0 -1  0  0  1  1  0  0
                0  0  0  0  1  1  1  0 -1 -1  0
                0  0  0  0  0  1  1  0  0  0  0
                0  0  0  1  0  1  0  0  0  0  0
                0  0 -1  0  0  0  0 -1 -1  0  1
               -1  1  0  0  0  0  0  0  0  0 -1]
        @test TotalUnimodularity._tu_partition(M12)
        @test tu_both_routes(M12)
    end

    # Matrices far beyond the exponential test's reach are decided by
    # splitting off 2-sum pieces first. A k-fold chain of K33 / K33ᵀ-type
    # blocks is TU; gluing on a block that contains a non-TU 3×3 keeps that
    # 3×3 as a submatrix, so the result is not TU.
    @testset "2-sum splitting of large matrices" begin
        function chain(k)
            M = K33
            for i in 1:k
                M = two_sum(M, isodd(i) ? K33dual_twosum : K33)
            end
            M
        end
        bad = [1 1 1; 1 1 0; 1 0 1; 0 1 1]      # first row is the glue row
        @test !naive_is_totally_unimodular(bad[2:end, :])
        for k in (5, 12, 40, 200)
            M = chain(k)
            @test minimum(size(M)) > 20
            @test is_totally_unimodular(M)
            @test is_totally_unimodular(Matrix{Int}(M'))
            @test !is_totally_unimodular(two_sum(M, bad))
            perm_r = randperm(MersenneTwister(k), size(M, 1))
            perm_c = randperm(MersenneTwister(k + 1), size(M, 2))
            @test is_totally_unimodular(M[perm_r, perm_c])
        end
    end

    # Blocks with no 2-separation that are neither network matrices nor
    # transposes of one are decided through a 3-separation. The family used
    # here is the 3-sum of the network matrix of K_n with the transpose of
    # the network matrix of K_m plus a vertex of degree 3; from 28×26 on it
    # is out of reach of the exponential test.
    @testset "3-separation route" begin
        # Network matrix of K_n (star tree at vertex 1, rows = arcs (1,v))
        # plus a parallel copy of arc (1,2), arranged as [Am a a; cᵀ 0 1].
        function A_Kn(n)
            cols = Vector{Int}[]
            for u in 2:n, v in u+1:n
                (u, v) == (2, n) && continue
                x = zeros(Int, n - 1); x[u-1] = 1; x[v-1] = -1; push!(cols, x)
            end
            x = zeros(Int, n - 1); x[1] = -1; push!(cols, x)
            y = copy(x); y[n-1] = 1; push!(cols, y)
            reduce(hcat, cols)
        end
        # Transpose of the network matrix of K_m plus a vertex w joined to
        # 1, 2, 3 (tree: 1→w, w→2, 1→v for v ≥ 3), as [1 0 bᵀ; d d Bm].
        function B_Km(m)
            cols = Vector{Int}[]
            x = zeros(Int, m); x[1] = 1; x[3] = -1; push!(cols, x)
            x = zeros(Int, m); x[1] = 1; x[2] = 1; push!(cols, x)
            for v in 3:m; x = zeros(Int, m); x[1] = 1; x[2] = 1; x[v] = -1; push!(cols, x); end
            for u in 3:m, v in u+1:m; x = zeros(Int, m); x[u] = -1; x[v] = 1; push!(cols, x); end
            Matrix{Int}(reduce(hcat, cols)')
        end
        rng = MersenneTwister(7)
        scramble(M) = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))] .*
                      rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2))
        function rand_pivot(M)
            p = rand(rng, findall(!iszero, M))
            pivot(M[[p[1]; setdiff(1:size(M, 1), p[1])],
                    [p[2]; setdiff(1:size(M, 2), p[2])]], 1)
        end

        for (n, m) in ((5, 5), (6, 6), (7, 7), (8, 8), (9, 9), (12, 12))
            M = scramble(three_sum(A_Kn(n), B_Km(m)))
            @test is_totally_unimodular(M)
            @test is_totally_unimodular(Matrix{Int}(M'))
            # Pivoting keeps a TU matrix TU and changes which kind of
            # 3-separation the matrix shows.
            P = M
            for _ in 1:4; P = rand_pivot(P); end
            @test is_totally_unimodular(P)
            # A non-TU 3×3 placed inside Bm survives the 3-sum as a submatrix.
            Bbad = B_Km(m)
            Bbad[2:4, 3:5] = [1 1 0; 1 0 1; 0 1 1]
            @test !is_totally_unimodular(scramble(three_sum(A_Kn(n), Bbad)))
        end

        # A non-TU matrix on which the CMR library (commit 1c1c6eaf, default
        # options) aborts with an internal error instead of answering; found
        # by fuzzing against it. Rows 1, 6, 7 and columns 1, 4, 6 have
        # determinant -2.
        cmr_abort = [-1  0  0  1  0  0  1
                      0  0 -1  0 -1  0  0
                      1  1  0 -1  1  0  0
                      0  0  1  0  0  0 -1
                      0  1  0  0  0  0  1
                      0  0  0 -1  0 -1  0
                      1  0  0  0  0 -1  0]
        @test !naive_is_totally_unimodular(cmr_abort)
        @test !tu_both_routes(cmr_abort)
        @test !tu_both_routes(Matrix{Int}(cmr_abort'))
        for alg in (:eulerian, :partition)
            @test !cmr_is_totally_unimodular(cmr_abort; algorithm = alg)
        end

        # Against the exact test, with the search route forced on at every
        # size: pivots, deletions and entry flips of the small members.
        base = [scramble(three_sum(A_Kn(n), B_Km(m))) for (n, m) in ((5, 5), (6, 5), (6, 6))]
        old = TotalUnimodularity._PARTITION_MAX_DIM[]
        TotalUnimodularity._PARTITION_MAX_DIM[] = 0
        n_bad = 0
        n_tu = 0
        n_total = 1500
        try
            for trial in 1:n_total
                M = rand(rng, base)
                for _ in 1:rand(rng, 0:6); M = rand_pivot(M); end
                for _ in 1:rand(rng, 0:3)
                    size(M, 1) > 5 && rand(rng, Bool) &&
                        (M = M[setdiff(1:size(M, 1), rand(rng, 1:size(M, 1))), :])
                    size(M, 2) > 5 && rand(rng, Bool) &&
                        (M = M[:, setdiff(1:size(M, 2), rand(rng, 1:size(M, 2)))])
                end
                M = scramble(M)
                if rand(rng) < 0.4
                    k = rand(rng, 1:length(M))
                    M[k] = rand(rng, setdiff(-1:1, M[k]))
                end
                want = TotalUnimodularity._tu_partition(M)
                n_tu += want
                if is_totally_unimodular(M) != want
                    n_bad += 1
                    @warn "DISAGREEMENT" trial M want
                end
            end
        finally
            TotalUnimodularity._PARTITION_MAX_DIM[] = old
        end
        @test n_bad == 0
        @test n_total ÷ 4 < n_tu < 3 * n_total ÷ 4
    end

    # An exception inside is_totally_unimodular fails these tests: it is a
    # predicate and must return an answer for every {-1,0,1} matrix.
    @testset "is_totally_unimodular vs naive (random, extended)" begin
        rng = MersenneTwister(123)
        n_bad = 0
        for trial in 1:2000
            M = rand(rng, (-1, 0, 1), rand(rng, 2:5), rand(rng, 2:6))
            naive = naive_is_totally_unimodular(M)
            fast = is_totally_unimodular(M)
            decomp = cmr_is_totally_unimodular(M; algorithm=:decomposition)
            if naive != fast || naive != decomp
                n_bad += 1
                @warn "DISAGREEMENT" trial M naive fast decomp
            end
        end
        @test n_bad == 0
    end

    # Oracle is the exact Ghouila-Houri test (itself checked against the
    # naive oracle in internals.jl); naive is too slow at these sizes.
    @testset "is_totally_unimodular vs partition (larger random)" begin
        rng = MersenneTwister(456)
        n_bad = 0
        for trial in 1:200
            M = rand(rng, (-1, 0, 1), rand(rng, 5:8), rand(rng, 5:10))
            want = TotalUnimodularity._tu_partition(M)
            fast = is_totally_unimodular(M)
            decomp = cmr_is_totally_unimodular(M; algorithm=:decomposition)
            if want != fast || want != decomp
                n_bad += 1
                @warn "DISAGREEMENT" trial M want fast decomp
            end
        end
        @test n_bad == 0
    end

    # Uniform random matrices almost never reach the decomposition cases
    # (they die at the Eulerian pre-filter or succeed as network matrices).
    # Compose structured inputs instead: 1-/2-sums of F_1, F_2, K33, K33ᵀ and
    # random network matrices, with random permutations, ±1 scalings and
    # pivots (all TU-preserving), then optionally flip one entry so that
    # roughly a third of the inputs are non-TU. Density-biased draws cover
    # the Ghouila-Houri-style searches.
    @testset "is_totally_unimodular vs partition (structured fuzz)" begin
        # Seed chosen so the stream includes inputs that hit the pivot-cycle
        # false negative fixed alongside "pivot cycles on TU matrices" (3 of
        # 2000 on the pre-fix code; some seeds hit none).
        rng = MersenneTwister(2)

        function rand_network(k, n)
            # Random tree on vertices 1..k+1 (tree arc v-1 joins parent[v], v);
            # column j is the signed tree path between two random vertices.
            parent = [0; [rand(rng, 1:v-1) for v in 2:k+1]]
            orient = rand(rng, (-1, 1), k)
            M = zeros(Int, k, n)
            for j in 1:n, (v, s) in ((rand(rng, 1:k+1), 1), (rand(rng, 1:k+1), -1))
                while v != 1
                    M[v-1, j] += s * orient[v-1]
                    v = parent[v]
                end
            end
            M
        end
        function scramble(M)
            M = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))]
            M .* rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2))
        end
        function rand_pivot(M)
            p = rand(rng, findall(!iszero, M))
            pivot(M[[p[1]; setdiff(1:size(M, 1), p[1])],
                    [p[2]; setdiff(1:size(M, 2), p[2])]], 1)
        end
        function block()
            c = rand(rng, 1:6)
            scramble(c == 1 ? F_1 : c == 2 ? F_2 : c == 3 ? K33 :
                     c == 4 ? Matrix{Int}(K33') :
                     rand_network(rand(rng, 2:5), rand(rng, 2:6)))
        end
        function composed()
            M = block()
            for _ in 1:rand(rng, 0:2)
                B = block()
                size(M, 1) + size(B, 1) > 13 && break
                M = scramble(rand(rng) < 0.8 ? two_sum(M, B) : one_sum(M, B))
                for _ in 1:rand(rng, 0:3)
                    iszero(M) || (M = rand_pivot(M))
                end
            end
            r, c = min(size(M, 1), 12), min(size(M, 2), 12)
            M = M[randperm(rng, size(M, 1))[1:r], randperm(rng, size(M, 2))[1:c]]
            if rand(rng) < 0.35
                i, j = rand(rng, 1:r), rand(rng, 1:c)
                M[i, j] = rand(rng, setdiff(-1:1, M[i, j]))
            end
            M
        end
        function biased()
            p = 0.2 + 0.7rand(rng)
            [rand(rng) < p ? rand(rng, (-1, 1)) : 0
             for _ in 1:rand(rng, 6:10), _ in 1:rand(rng, 6:10)]
        end

        n_bad = 0
        n_tu = 0
        n_total = 2000
        for trial in 1:n_total
            M = rand(rng) < 0.85 ? composed() : biased()
            want = TotalUnimodularity._tu_partition(M)
            fast = is_totally_unimodular(M)
            decomp = cmr_is_totally_unimodular(M; algorithm=:decomposition)
            n_tu += want
            if want != fast || want != decomp
                n_bad += 1
                @warn "DISAGREEMENT" trial M want fast decomp
            end
        end
        @test n_bad == 0
        # The generator must keep producing both answers in bulk.
        @test n_total ÷ 4 < n_tu < 3 * n_total ÷ 4
    end


    # The decomposition is expected never to give up and hand a matrix to
    # the exact test (see _FALLBACKS). Everything above ran without doing so.
    @testset "no fallback to the exact test" begin
        @test map(x -> x[], TotalUnimodularity._FALLBACKS) == fallbacks_before
    end

end
