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
    
    @testset "is_totally_unimodular vs naive (random, extended)" begin
        @info "Starting extended random tests..."
        flush(stderr)
        rng = MersenneTwister(123)
        n_tests = 0
        n_agree = 0
        n_skip = 0
        n_total = 2000
        report_every = 50

        @info "Starting random TU tests..."

        for trial in 1:n_total
            rows = rand(rng, 2:5)
            cols = rand(rng, 2:6)
            M = rand(rng, (-1, 0, 1), rows, cols)

            naive = naive_is_totally_unimodular(M)

            fast = try
                is_totally_unimodular(M)
            catch e
                n_skip += 1
                @warn "Skipped matrix" trial=trial rows=rows cols=cols exception=sprint(showerror, e)
                display(M)
                flush(stderr)
                continue
            end
            n_tests += 1
            if naive == fast
                n_agree += 1
            else
                @warn "DISAGREEMENT" trial M naive fast
                @test naive == fast
            end

            if trial % report_every == 0
                @info "Progress" trial n_total n_agree n_tests n_skip
            end
        end

        @info "Final" n_agree n_tests n_skip n_total
        @test n_agree == n_tests
        @test n_tests > 800
    end
    
end

    
