using TotalUnimodularity
using Test
using LinearAlgebra

include("internals.jl")

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

        # TODO: add more tests once Cases 5-6 implemented
    end
    
end

    
