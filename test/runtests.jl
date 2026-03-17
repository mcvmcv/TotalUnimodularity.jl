using TotalUnimodularity
using Test
using LinearAlgebra

@testset "TotalUnimodularity.jl" begin
    @testset "Known TU matrices" begin
        # Identity matrix is TU
        @test naive_is_totally_unimodular(Matrix{Int}(I, 3, 3))

        # F_1 and F_2 are not TU
        @test naive_is_totally_unimodular(F_1)
        @test naive_is_totally_unimodular(F_2)

        # Simple network matrix is TU
        @test naive_is_totally_unimodular([1 0 1; 0 1 1; -1 0 0])

	# This matrices are not TU
        @test !naive_is_totally_unimodular([1 0 1; 1 1 0; 0 1 1])
	@test !naive_is_totally_unimodular([1 0 0 1; 1 0 1 0; 1 1 0 0; 1 1 1 1])
    end
end
    
