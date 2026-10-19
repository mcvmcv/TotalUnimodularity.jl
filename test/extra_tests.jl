# Extra tests from a pre-registration review (October 2026), included at the
# end of runtests.jl. The thread-safety test only means something with
# several threads: `Pkg.test("TotalUnimodularity"; julia_args = ["--threads=4"])`.
#
# Every expected answer here comes from theory, not from running the package:
#   * incidence matrices of directed graphs are TU;
#   * incidence matrices of undirected graphs are TU iff the graph is bipartite;
#   * matrices with the consecutive-ones property in each row are TU;
#   * the incidence matrix of an odd cycle has determinant ±2, and every proper
#     square submatrix of it is TU (it is minimally non-TU), so for k ≥ 5 the
#     2×2 / 3×3 pre-filter cannot see the violation;
#   * TU is preserved by transposition, permutations, ±1 scalings, pivoting on
#     a ±1 entry, and by appending copies or negations of rows or columns,
#     unit rows or columns, or an identity block — and each of these
#     operations also preserves non-TU-ness.

using TotalUnimodularity
using Test
using Random
using LinearAlgebra
using SparseArrays
using Aqua

const XRNG = MersenneTwister(20261020)
const xfallbacks_before = map(x -> x[], TotalUnimodularity._FALLBACKS)

# ── Generators ────────────────────────────────────────────────────────────────

# Node–arc incidence matrix of a random digraph on n nodes with m arcs.
function digraph_incidence(n, m; rng = XRNG)
    A = zeros(Int, n, m)
    for j in 1:m
        u, v = randperm(rng, n)[1:2]
        A[u, j] = 1
        A[v, j] = -1
    end
    A
end

# Vertex–edge incidence matrix of a random bipartite graph with sides of size
# p and q and m edges (rows 1:p are one side, p+1:p+q the other).
function bipartite_incidence(p, q, m; rng = XRNG)
    A = zeros(Int, p + q, m)
    for j in 1:m
        A[rand(rng, 1:p), j] = 1
        A[p + rand(rng, 1:q), j] = 1
    end
    A
end

# Each row is a random interval of consecutive ones.
function interval_matrix(m, n; rng = XRNG)
    A = zeros(Int, m, n)
    for i in 1:m
        a = rand(rng, 1:n)
        b = rand(rng, a:n)
        A[i, a:b] .= 1
    end
    A
end

# Vertex–edge incidence matrix of the cycle C_k (k×k).
function cycle_incidence(k)
    A = zeros(Int, k, k)
    for j in 1:k
        A[j, j] = 1
        A[mod1(j + 1, k), j] = 1
    end
    A
end

# A connected bipartite graph on 2p vertices (a spanning zig-zag path plus p
# random edges), then one edge inside a side, which closes an odd cycle.
function bipartite_plus_odd_edge(p; rng = XRNG)
    edges = Tuple{Int,Int}[]
    for i in 1:p
        push!(edges, (i, p + i))
        i < p && push!(edges, (i + 1, p + i))
    end
    for _ in 1:p
        push!(edges, (rand(rng, 1:p), p + rand(rng, 1:p)))
    end
    push!(edges, (1, p))                       # both on the left side
    A = zeros(Int, 2p, length(edges))
    for (j, (u, v)) in enumerate(edges)
        A[u, j] = 1
        A[v, j] = 1
    end
    A
end

# Random row/column permutation and ±1 scaling.
function scramble(M; rng = XRNG)
    M = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))]
    M .* rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2))
end

# Pivot on a random nonzero entry (which is ±1 for the inputs used here).
function rand_pivot(M; rng = XRNG)
    p = rand(rng, findall(!iszero, M))
    r = [p[1]; setdiff(1:size(M, 1), p[1])]
    c = [p[2]; setdiff(1:size(M, 2), p[2])]
    pivot(M[r, c], 1)
end

# Matrices that are TU iff M is.
function tu_equivalent_variants(M; rng = XRNG)
    m, n = size(M)
    Iₘ = Matrix{Int}(I, m, m)
    e = zeros(Int, 1, n); e[rand(rng, 1:n)] = 1
    V = Any[
        Matrix{Int}(M'),
        scramble(M; rng),
        [M Iₘ],
        [M -M],
        [M; M],
        [M; -M],
        [M; e],
        [M M[:, rand(rng, 1:n)]],
    ]
    # One pivot, only while the entries stay in {-1, 0, 1}.
    if any(!iszero, M)
        P = rand_pivot(M; rng)
        all(x -> -1 <= x <= 1, P) && push!(V, P)
    end
    V
end

@testset "Extra tests (pre-registration review)" verbose = true begin

    @testset "degenerate shapes" begin
        for alg in (:decomposition, :exhaustive, :eulerian, :partition, :naive)
            @test is_totally_unimodular(zeros(Int, 0, 0); algorithm = alg)
            @test is_totally_unimodular(zeros(Int, 0, 4); algorithm = alg)
            @test is_totally_unimodular(zeros(Int, 4, 0); algorithm = alg)
            @test is_totally_unimodular(zeros(Int, 5, 7); algorithm = alg)
            @test is_totally_unimodular([1 -1 1 0 1]; algorithm = alg)
            @test is_totally_unimodular(reshape([1, -1, 0, 1], 4, 1); algorithm = alg)
            for x in (-1, 0, 1)
                @test is_totally_unimodular(fill(x, 1, 1); algorithm = alg)
            end
            for x in (-2, 2, 3, typemin(Int), typemax(Int))
                @test !is_totally_unimodular(fill(x, 1, 1); algorithm = alg)
            end
        end
    end

    @testset "input types" begin
        M = interval_matrix(6, 8)
        bad = cycle_incidence(5)
        @test is_totally_unimodular(sparse(M))
        @test !is_totally_unimodular(sparse(bad))
        @test is_totally_unimodular(UInt8.(M))
        @test !is_totally_unimodular(UInt8[2 0; 0 1])
        @test is_totally_unimodular(BigInt.(M))
        @test !is_totally_unimodular(BigInt.(bad))
        @test is_totally_unimodular(Bool.(M))
        @test is_totally_unimodular(view(M, 2:5, :))
        @test is_totally_unimodular(@view M[:, end:-1:1])
        # The input must not be modified.
        M0 = copy(M)
        is_totally_unimodular(M)
        @test M == M0
        # Real matrices, as optimisation models store them (e.g. sparse
        # Float64 from JuMP): integer-valued entries are converted, and any
        # entry outside {-1, 0, 1} — non-integer or NaN included — is a bad
        # 1×1 submatrix.
        @test is_totally_unimodular(float.(M))
        @test !is_totally_unimodular(float.(bad))
        @test is_totally_unimodular(sparse(float.(M)))
        @test is_totally_unimodular(Rational.(M))
        @test is_totally_unimodular(Float32.(M))
        @test is_totally_unimodular([-0.0 1.0; 1.0 1.0])
        @test !is_totally_unimodular([0.5 0; 0 1])
        @test !is_totally_unimodular([1 // 2 0; 0 1])
        @test !is_totally_unimodular([NaN 0; 0 1])
        @test !is_totally_unimodular([Inf 0; 0 1])
        @test !is_totally_unimodular([1.0 1.0; -1.0 1.0])   # det 2
        for alg in (:decomposition, :exhaustive, :eulerian, :partition, :naive)
            @test is_totally_unimodular(float.(M); algorithm = alg)
            @test !is_totally_unimodular([0.5 0; 0 1]; algorithm = alg)
        end
        @test_throws ArgumentError is_totally_unimodular([0.5 0; 0 1]; algorithm = :bogus)
    end

    @testset "graph families with known answers" begin
        for (n, m) in ((5, 8), (20, 60), (60, 200), (150, 600))
            @test is_totally_unimodular(digraph_incidence(n, m))
            @test is_totally_unimodular(Matrix{Int}(digraph_incidence(n, m)'))
        end
        for (p, q, m) in ((4, 5, 10), (20, 25, 80), (80, 70, 400))
            @test is_totally_unimodular(bipartite_incidence(p, q, m))
        end
        for (m, n) in ((6, 8), (40, 30), (200, 150))
            @test is_totally_unimodular(interval_matrix(m, n))
            @test is_totally_unimodular(Matrix{Int}(interval_matrix(m, n)'))
        end
    end

    @testset "minimally non-TU: odd cycles" begin
        for k in (3, 5, 7, 9)
            @test !is_totally_unimodular(cycle_incidence(k); algorithm = :naive)
        end
        for k in (3, 5, 7, 9, 15, 31, 61, 101)
            C = cycle_incidence(k)
            @test !is_totally_unimodular(C)
            @test !is_totally_unimodular(scramble(C))
            @test !is_totally_unimodular(Matrix{Int}(C'))
        end
        for k in (4, 6, 10, 30, 100)          # even cycles are bipartite
            @test is_totally_unimodular(cycle_incidence(k))
            @test is_totally_unimodular(scramble(cycle_incidence(k)))
        end
        # An odd cycle hidden in a large connected bipartite graph.
        for p in (4, 10, 40, 120)
            G = bipartite_plus_odd_edge(p)
            @test !is_totally_unimodular(G)
            @test !is_totally_unimodular(scramble(G))
            @test is_totally_unimodular(G[:, 1:end-1])   # without the odd edge
        end
    end

    @testset "metamorphic: invariance under TU-preserving operations" begin
        base_tu = Any[F_1, F_2, digraph_incidence(30, 80), interval_matrix(40, 35),
                      bipartite_incidence(15, 20, 60)]
        base_bad = Any[cycle_incidence(9), cycle_incidence(41),
                       bipartite_plus_odd_edge(25)]
        for M in base_tu, V in tu_equivalent_variants(M)
            @test is_totally_unimodular(V)
        end
        for M in base_bad, V in tu_equivalent_variants(M)
            @test !is_totally_unimodular(V)
        end
        # Long random walks of pivots and scramblings, compared with the
        # exact test on sizes it can handle.
        n_bad = 0
        for trial in 1:300
            M = rand(XRNG) < 0.5 ? digraph_incidence(rand(XRNG, 4:7), rand(XRNG, 5:9)) :
                                   interval_matrix(rand(XRNG, 4:7), rand(XRNG, 5:9))
            rand(XRNG) < 0.4 && (M[rand(XRNG, 1:length(M))] = rand(XRNG, (-1, 1)))
            for _ in 1:rand(XRNG, 0:8)
                iszero(M) && break
                P = rand_pivot(M)
                all(x -> -1 <= x <= 1, P) || break
                M = scramble(P)
            end
            want = TotalUnimodularity._tu_partition(M)
            if is_totally_unimodular(M) != want
                n_bad += 1
                @warn "DISAGREEMENT" trial M want
            end
        end
        @test n_bad == 0
    end

    @testset "thread safety" begin
        inputs = Any[]
        for _ in 1:40
            push!(inputs, digraph_incidence(20, 50))
            push!(inputs, cycle_incidence(rand(XRNG, (11, 13, 15))))
            push!(inputs, scramble(bipartite_plus_odd_edge(12)))
        end
        serial = [is_totally_unimodular(M) for M in inputs]
        parallel = Vector{Bool}(undef, length(inputs))
        Threads.@threads for i in eachindex(inputs)
            parallel[i] = is_totally_unimodular(inputs[i])
        end
        @test parallel == serial
        @test count(serial) == 40
    end

    @testset "no fallback to the exact test" begin
        @test map(x -> x[], TotalUnimodularity._FALLBACKS) == xfallbacks_before
    end

    @testset "Aqua" begin
        Aqua.test_all(TotalUnimodularity)
    end
end
