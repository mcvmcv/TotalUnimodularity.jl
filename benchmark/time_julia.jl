# Times is_totally_unimodular on one matrix file (CMR dense format) and prints
# "<answer>\t<seconds>". Further arguments are matrix files run first as JIT
# warmup.
#
#     julia --project=. benchmark/time_julia.jl MATRIX [WARMUP...]

using TotalUnimodularity

function readmat(f)
    L = [l for l in readlines(f)[2:end] if !isempty(strip(l))]
    permutedims(reduce(hcat, [parse.(Int, split(l)) for l in L]))
end

function main()
    M = readmat(ARGS[1])
    for f in ARGS[2:end]; is_totally_unimodular(readmat(f)); end
    t = @elapsed r = is_totally_unimodular(M)
    # Best of 21 runs under 50 ms, best of 4 up to 2 s, a single run above.
    if t < 2
        for _ in 1:(t < 0.05 ? 20 : 3); t = min(t, @elapsed is_totally_unimodular(M)); end
    end
    println(r, "\t", t)
end
main()
