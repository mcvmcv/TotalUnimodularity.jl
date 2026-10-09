# Differential fuzz of is_totally_unimodular against the CMR library on
# matrices too large for the exact test: random pivots, row and column
# deletions and entry changes applied to the 3-sum family written by
# generate.jl (21×19 to 66×64). The transpose of every input is checked too.
#
#     julia --project=. benchmark/generate.jl OUTDIR
#     CMR_TU=/path/to/cmr/build/cmr-tu julia --project=. benchmark/fuzz_cmr.jl OUTDIR [N [SEED...]]
#
# N inputs per seed (default 1500, seeds 1 2 3). Prints one line per seed and
# every disagreement in CMR's dense format; exits with status 1 if there was
# one. Not part of the test suite because it needs the CMR binary.

using TotalUnimodularity, Random

const CMR_TU = get(ENV, "CMR_TU", "cmr-tu")

function readmat(f)
    L = [l for l in readlines(f)[2:end] if !isempty(strip(l))]
    permutedims(reduce(hcat, [parse.(Int, split(l)) for l in L]))
end

function writemat(io, M)
    println(io, size(M, 1), " ", size(M, 2))
    for r in eachrow(M); println(io, join(r, " ")); end
end

# CMR's answer, or `nothing` if it gives none. Up to commit 1c1c6eaf, cmr-tu
# with default options aborts on some non-TU matrices ("User input error",
# discopt/cmr issue #115), so a failed run is repeated without the simple
# separation search, which avoids that code path. The fix for that issue
# renames the option, so both spellings are tried.
function cmr(M)
    f = tempname()
    open(io -> writemat(io, M), f, "w")
    try
        for opts in (String[], ["--no-simple-3-sepa"], ["--no-simple-sepa"])
            out = try read(pipeline(`$CMR_TU $f $opts`; stderr = devnull), String) catch; "" end
            occursin("IS totally", out) && return true
            occursin("IS NOT", out) && return false
        end
        return nothing
    finally
        rm(f; force = true)
    end
end

function main()
    dir = ARGS[1]
    N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 1500
    seeds = length(ARGS) >= 3 ? parse.(Int, ARGS[3:end]) : [1, 2, 3]

    # The TU members of the family with at least 19 rows and columns.
    base = Matrix{Int}[]
    for line in eachline(joinpath(dir, "manifest.tsv"))
        f, name, _ = split(line, '\t')
        (occursin("⊕₃", name) && !occursin("flipped", name)) || continue
        M = readmat(f)
        minimum(size(M)) >= 19 && push!(base, M)
    end
    isempty(base) && error("no 3-sum matrices found in $dir; run benchmark/generate.jl first")
    cmr(base[1]) === true || error("`$CMR_TU` did not report the first base matrix as TU; set CMR_TU")
    is_totally_unimodular(base[1])                 # JIT warmup

    total_bad = 0
    for seed in seeds
        rng = MersenneTwister(seed)
        scramble(M) = (M = M[randperm(rng, size(M, 1)), randperm(rng, size(M, 2))];
                       M .* rand(rng, (-1, 1), size(M, 1)) .* rand(rng, (-1, 1), 1, size(M, 2)))
        function rand_pivot(M)
            p = rand(rng, findall(!iszero, M))
            pivot(M[[p[1]; setdiff(1:size(M, 1), p[1])], [p[2]; setdiff(1:size(M, 2), p[2])]], 1)
        end
        bad = 0; n_tu = 0; unanswered = 0; slowest = 0.0
        for trial in 1:N
            M = rand(rng, base)
            for _ in 1:rand(rng, 0:8); M = rand_pivot(M); end
            for _ in 1:rand(rng, 0:4)
                size(M, 1) > 5 && rand(rng, Bool) && (M = M[setdiff(1:size(M, 1), rand(rng, 1:size(M, 1))), :])
                size(M, 2) > 5 && rand(rng, Bool) && (M = M[:, setdiff(1:size(M, 2), rand(rng, 1:size(M, 2)))])
            end
            M = scramble(M)
            if rand(rng) < 0.4
                for _ in 1:rand(rng, 1:2)
                    k = rand(rng, 1:length(M))
                    M[k] = rand(rng, setdiff(-1:1, M[k]))
                end
            end
            want = cmr(M)
            t = @elapsed got = is_totally_unimodular(M)
            got_t = is_totally_unimodular(Matrix{Int}(M'))
            slowest = max(slowest, t)
            if want === nothing
                unanswered += 1
                want = got                          # still compare with the transpose
            end
            n_tu += want
            if got != want || got_t != want
                bad += 1
                println("DISAGREEMENT seed=$seed trial=$trial cmr=$want julia=$got julia(transpose)=$got_t")
                writemat(stdout, M)
            end
        end
        total_bad += bad
        println("seed=$seed n=$N tu=$n_tu disagreements=$bad cmr-unanswered=$unanswered ",
                "slowest=$(round(slowest, sigdigits = 3))s")
        flush(stdout)
    end
    fb = map(x -> x[], TotalUnimodularity._FALLBACKS)
    println("fallbacks to the exact test: ", fb)
    exit(total_bad == 0 ? 0 : 1)
end

main()
