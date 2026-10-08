# Writes the benchmark matrices of the README's "CMR vs is_totally_unimodular"
# tables to a directory, in CMR's dense format ("m n" header, then the rows),
# together with manifest.tsv (file, name, size).
#
#     julia --project=. benchmark/generate.jl OUTDIR
#
# Run from the repository root: the CMR test matrices are read from
# test/test_cmr.jl. The random matrices are seeded, so every run writes the
# same files.

using TotalUnimodularity, Random

outdir = ARGS[1]
mkpath(outdir)

# Matrix literals of test/test_cmr.jl, in file order.
src = read(joinpath("test", "test_cmr.jl"), String)
lits = Matrix{Int}[]
for m in eachmatch(r"\[\s*\n((?:[ \t]*-?[01](?:[ \t]+-?[01])*[ \t]*\n)+)[ \t]*\]", src)
    rows = [parse.(Int, split(l)) for l in split(strip(m.captures[1]), "\n")]
    push!(lits, permutedims(reduce(hcat, rows)))
end
bysize(r, c) = lits[findfirst(M -> size(M) == (r, c), lits)]
K33 = bysize(5, 4)
K33d = lits[2]                      # K33dual_twosum
R10 = [1 0 0 1 1; 1 1 0 0 1; 0 1 1 0 1; 0 0 1 1 1; 1 1 1 1 1]
R12 = [1  0 1  1 0 0
       0  1 1  1 0 0
       1  0 1  0 1 1
       0 -1 0 -1 1 1
       1  0 1  0 1 0
       0 -1 0 -1 0 1]

scramble(rng, M) = (M = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))];
                    M .* rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2)))

# Random network matrix: a random tree on vertices 1..k+1 (tree arc v-1 joins
# parent[v] and v); each column is the signed tree path between two vertices.
function rand_network(rng, k, n)
    parent = [0; [rand(rng, 1:v-1) for v in 2:k+1]]
    orient = rand(rng, (-1, 1), k)
    function up(v)
        x = zeros(Int, k)
        while v != 1; x[v-1] = orient[v-1]; v = parent[v]; end
        x
    end
    M = zeros(Int, k, n)
    for j in 1:n
        u = rand(rng, 1:k+1); v = rand(rng, setdiff(1:k+1, u))
        M[:, j] = up(u) - up(v)
    end
    M
end

function chain(k)
    M = K33
    for i in 1:k; M = two_sum(M, isodd(i) ? K33d : K33); end
    M
end
bad = [1 1 1; 1 1 0; 1 0 1; 0 1 1]      # rows 2:4 are a non-TU 3×3

# Network matrix of K_n (star tree at vertex 1, rows = arcs (1,v), v = 2..n)
# plus a parallel copy of arc (1,2), arranged as [Am a a; cᵀ 0 1] for three_sum.
function A_Kn(n)
    row(v) = v - 1
    cols = Vector{Int}[]
    for u in 2:n, v in u+1:n
        (u, v) == (2, n) && continue
        x = zeros(Int, n - 1); x[row(u)] = 1; x[row(v)] = -1; push!(cols, x)
    end
    x = zeros(Int, n - 1); x[row(2)] = -1; push!(cols, x)
    y = copy(x); y[row(n)] = 1; push!(cols, y)
    reduce(hcat, cols)
end

# Transpose of the network matrix of K_m plus a vertex w joined to 1, 2, 3
# (tree: 1→w, w→2, 1→v for v ≥ 3), arranged as [1 0 bᵀ; d d Bm] for three_sum.
function B_Km(m)
    cols = Vector{Int}[]        # rows: 1 = (1,w), 2 = (w,2), v = (1,v) for v ≥ 3
    x = zeros(Int, m); x[1] = 1; x[3] = -1; push!(cols, x)             # (3,w)
    x = zeros(Int, m); x[1] = 1; x[2] = 1; push!(cols, x)              # (1,2)
    for v in 3:m; x = zeros(Int, m); x[1] = 1; x[2] = 1; x[v] = -1; push!(cols, x); end
    for u in 3:m, v in u+1:m; x = zeros(Int, m); x[u] = -1; x[v] = 1; push!(cols, x); end
    Matrix{Int}(reduce(hcat, cols)')
end

cases = Pair{String,Matrix{Int}}[]
add(name, M) = push!(cases, name => M)

add("K33", K33); add("F_1", F_1); add("R10", R10); add("R12", R12)
add("Fano", bysize(3, 4))
add("CMR Eulerian test", bysize(12, 12))
add("CMR partition test", bysize(14, 14))
for k in (1, 3, 5, 12, 40, 200); add("2-sum chain, $(k+1) blocks", chain(k)); end
for k in (12, 200); add("2-sum chain ($(k+1) blocks) + non-TU block", two_sum(chain(k), bad)); end

let rng = MersenneTwister(2026)
    for (k, n) in ((20, 40), (50, 100), (100, 200), (200, 400))
        add("random network", scramble(rng, rand_network(rng, k, n)))
    end
    M = scramble(rng, rand_network(rng, 50, 100))
    M[rand(rng, findall(iszero, M))] = 1
    add("network, one entry flipped", M)
    for (s, p) in ((30, 0.1), (100, 0.05), (20, 0.5))
        add("random sparse, density $p",
            [rand(rng) < p ? rand(rng, (-1, 1)) : 0 for _ in 1:s, _ in 1:s])
    end
end

# No 2-separation, neither network nor co-network: the hard family.
let rng = MersenneTwister(7)
    for (n, m) in ((5, 5), (6, 5), (6, 6), (7, 6), (7, 7), (8, 7), (8, 8), (9, 9), (12, 12))
        M = scramble(rng, three_sum(A_Kn(n), B_Km(m)))
        N = copy(M); N[rand(rng, findall(iszero, N))] = 1
        add("K$n ⊕₃ K$m*", M)
        add("K$n ⊕₃ K$m*, one entry flipped", N)
    end
end

open(joinpath(outdir, "manifest.tsv"), "w") do mf
    for (i, (name, M)) in enumerate(cases)
        f = joinpath(outdir, "m$(lpad(i, 2, '0')).mat")
        open(f, "w") do io
            println(io, size(M, 1), " ", size(M, 2))
            for r in eachrow(M); println(io, join(r, " ")); end
        end
        println(mf, f, "\t", name, "\t", size(M, 1), "×", size(M, 2))
    end
end
