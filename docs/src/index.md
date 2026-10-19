# TotalUnimodularity.jl

A pure Julia test for total unimodularity: whether every square submatrix of
an integer matrix has determinant in {-1, 0, 1}. It follows the recognition
algorithm based on Seymour's decomposition theorem (Schrijver, *Theory of
Linear and Integer Programming*, Chapter 20), with polynomial-time searches
for the 2- and 3-separations it needs.

```jldoctest
julia> using TotalUnimodularity

julia> is_totally_unimodular([1 0 1 0; 0 1 1 0; 0 0 1 1])
true

julia> is_totally_unimodular([1 1 0; 1 0 1; 0 1 1])
false
```

Any `AbstractMatrix` of real numbers is accepted (integers, `Bool`,
`Float64`, `Rational`, sparse matrices, views); entries outside {-1, 0, 1}
give `false`. The `algorithm` keyword selects one of several
independent exact methods, used for verification; see
[`is_totally_unimodular`](@ref).

The [README](https://github.com/mcvmcv/TotalUnimodularity.jl#readme) has
benchmarks, including a comparison with the
[CMR](https://github.com/discopt/cmr) C library, and the repository's
`THEORY.md` and `IMPLEMENTATION_NOTES.md` explain the algorithm and the
implementation.
