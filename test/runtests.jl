using TotalUnimodularity
using Test
using LinearAlgebra
using Random

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
    
    @testset "is_totally_unimodular vs naive (random)" begin
        rng = MersenneTwister(42)
        n_tests = 0
        n_agree = 0
        n_skip = 0  # cases that hit unimplemented paths

        for _ in 1:200
            # Random small {-1,0,1} matrix
            rows = rand(rng, 2:6)
            cols = rand(rng, 2:6)
            M = rand(rng, (-1, 0, 1), rows, cols)

            naive = naive_is_totally_unimodular(M)
        
            fast = try
                is_totally_unimodular(M)
            catch e
                if e isa ErrorException && occursin("not yet implemented", e.msg)
                    n_skip += 1
                    continue
                end
                rethrow(e)
            end

            n_tests += 1
            if naive == fast
                n_agree += 1
            else
                @test naive == fast  # will print the failing matrix
                println("Disagreement on:")
                display(M)
                println("naive=$naive, fast=$fast")
            end
        end

        println("Tested $n_tests matrices, $n_agree agreed, $n_skip skipped")
        @test n_agree == n_tests  # all tested cases agree
        @test n_tests > 100       # make sure we tested enough
    end
    
end

    
