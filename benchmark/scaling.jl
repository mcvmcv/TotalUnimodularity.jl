# How is_totally_unimodular and CMR scale beyond the sizes of run.sh: members
# of the K_n ⊕₃ K_n* family (see generate.jl) from 36×34 to 378×376, each as
# built, after six random pivots, and with one entry changed so that the
# matrix is not TU but passes the Eulerian pre-filter (the case in which the
# 3-separation search has to try every candidate).
#
#     CMR_TU=/path/to/cmr/build/cmr-tu julia --project=. benchmark/scaling.jl
#
# CMR's time is the recognition time it reports with --stats; Julia's is the
# best of three runs under 2 s and a single run above. Takes a few minutes.

using TotalUnimodularity, Random
const T = TotalUnimodularity
const CMR = get(ENV, "CMR_TU", "cmr-tu")
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
function B_Km(m)
    cols = Vector{Int}[]
    x = zeros(Int, m); x[1] = 1; x[3] = -1; push!(cols, x)
    x = zeros(Int, m); x[1] = 1; x[2] = 1; push!(cols, x)
    for v in 3:m; x = zeros(Int, m); x[1] = 1; x[2] = 1; x[v] = -1; push!(cols, x); end
    for u in 3:m, v in u+1:m; x = zeros(Int, m); x[u] = -1; x[v] = 1; push!(cols, x); end
    Matrix{Int}(reduce(hcat, cols)')
end
function cmr(M)
    f = tempname(); open(f, "w") do io
        println(io, size(M, 1), " ", size(M, 2)); for r in eachrow(M); println(io, join(r, " ")); end
    end
    for opts in (String[], ["--no-simple-3-sepa"])
        out = try read(`bash -c "$CMR $f --stats $(join(opts, " ")) 2>&1"`, String) catch e; "" end
        ans = occursin("IS totally", out) ? true : occursin("IS NOT", out) ? false : nothing
        ans === nothing && continue
        t = 0.0
        for l in split(out, '\n'); m = match(r"(seymour|camion) total: \d+ in ([0-9.]+) seconds", l); m === nothing || (t += parse(Float64, m.captures[2])); end
        rm(f); return ans, t
    end
    rm(f); (nothing, NaN)
end
rng = MersenneTwister(11)
scramble(M) = (M = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))]; M .* rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2)))
function rand_pivot(M)
    p = rand(rng, findall(!iszero, M))
    pivot(M[[p[1]; setdiff(1:size(M, 1), p[1])], [p[2]; setdiff(1:size(M, 2), p[2])]], 1)
end
is_totally_unimodular(scramble(three_sum(A_Kn(6), B_Km(6))))
println(rpad("case", 26), rpad("size", 12), rpad("answer", 8), rpad("CMR", 12), rpad("Julia", 12), "Julia/CMR")
for n in (9, 12, 16, 20, 24, 28)
    base = scramble(three_sum(A_Kn(n), B_Km(n)))
    P = base; for _ in 1:6; P = rand_pivot(P); end; P = scramble(P)
    cases = Pair{String,Matrix{Int}}["TU" => base, "TU, pivoted" => P]
    # non-TU that survives the pre-filter: flip entries of the pivoted matrix until one does
    found = 0
    for _ in 1:400
        X = copy(P); k = rand(rng, 1:length(X)); X[k] = rand(rng, setdiff(-1:1, X[k]))
        T._tu_eulerian(X, 3; budget = 4 * prod(size(X)) * sum(size(X))) || continue
        a, _ = cmr(X); a === false || continue
        push!(cases, "non-TU, passes filter" => X); found += 1; found == 2 && break
    end
    for (name, M) in cases
        a, tc = cmr(M)
        tj = @elapsed r = is_totally_unimodular(M)
        tj < 2 && (tj = min(tj, @elapsed(is_totally_unimodular(M)), @elapsed(is_totally_unimodular(M))))
        println(rpad(name, 26), rpad(string(size(M)), 12), rpad(string(r), 8), rpad(string(round(tc * 1e3, sigdigits = 3), " ms"), 12), rpad(string(round(tj * 1e3, sigdigits = 3), " ms"), 12), round(tj / tc, sigdigits = 2), a == r ? "" : "  MISMATCH cmr=$a")
        flush(stdout)
    end
end
