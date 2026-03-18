using Random

import TotalUnimodularity: _is_standard_basis_vector, _is_trivial_vector,
                            _is_special_matrix, _drop_trivial_vectors,
                            _reduce_trivial_vectors, _has_dependent_rows,
                            _has_dependent_cols, _has_dependent_vectors,
                            _drop_dependent_rows, _drop_dependent_cols,
                            _drop_dependent_vectors, _reduce
@testset "Internals" begin

    @testset "_is_standard_basis_vector" begin
        @test _is_standard_basis_vector([1, 0, 0])
        @test _is_standard_basis_vector([0, 1, 0])
        @test !_is_standard_basis_vector([2, 0, 0])
        @test !_is_standard_basis_vector([0, 0, 0])
        @test !_is_standard_basis_vector([1, 1, 0])
    end

    @testset "_is_trivial_vector" begin
        @test _is_trivial_vector([0, 0, 0])
        @test _is_trivial_vector([1, 0, 0])
        @test !_is_trivial_vector([1, 1, 0])
        @test !_is_trivial_vector([2, 0, 0])
    end

    @testset "_is_special_matrix" begin
        # Recognise F_1 and F_2
        @test _is_special_matrix(F_1)
        @test _is_special_matrix(F_2)

        # Row/column permutations
        @test _is_special_matrix(F_1[[2,1,3,4,5], :])
        @test _is_special_matrix(F_1[:, [2,1,3,4,5]])
        @test _is_special_matrix(F_2[[2,1,3,4,5], :])
        @test _is_special_matrix(F_2[:, [2,1,3,4,5]])

        # ±1 row/column scalings
        @test _is_special_matrix([-F_1[1:1,:]; F_1[2:end,:]])
        @test _is_special_matrix([-F_2[1:1,:]; F_2[2:end,:]])

        # Non-special matrices
        @test !_is_special_matrix(Matrix{Int}(I, 5, 5))
        @test !_is_special_matrix(Matrix{Int}(I, 3, 3))
        @test !_is_special_matrix(F_1[1:4, 1:4])

    end

    @testset "_is_special_matrix random" begin
        rng = MersenneTwister(42)
        n = 5
        for _ in 1:100
            for target in (F_1, F_2)
                row_perm = randperm(rng, n)
                col_perm = randperm(rng, n)
                row_signs = rand(rng, (-1, 1), n)
                col_signs = rand(rng, (-1, 1), n)
                M = Diagonal(row_signs) * target[row_perm, col_perm] * Diagonal(col_signs)
                @test _is_special_matrix(M)
            end
        end
    end

    @testset "_is_special_matrix rejects non-special" begin
        rng = MersenneTwister(42)
        n = 5
        rejections = 0
        for _ in 1:1000
            M = rand(rng, (-1, 0, 1), n, n)
            # If not TU, definitely not F_1 or F_2
            if !naive_is_totally_unimodular(M)
                @test !_is_special_matrix(M)
                rejections += 1
            end
        end
        # Make sure we actually tested some cases
        @test rejections > 100
    end

    @testset "_has_dependent_rows" begin
        @test _has_dependent_rows([1 0 1; 1 0 1; 0 1 0])
        @test _has_dependent_rows([1 0 1; -1 0 -1; 0 1 0])
        @test !_has_dependent_rows([1 0 1; 0 1 1; 1 1 0])
        @test !_has_dependent_rows(F_1)
        @test !_has_dependent_rows(F_2)
    end

    @testset "_has_dependent_cols" begin
        @test _has_dependent_cols([1 1 0; 0 0 1; 1 1 0])
        @test _has_dependent_cols([1 -1 0; 0 0 1; 1 -1 0])
        @test !_has_dependent_cols([1 0 1; 0 1 1; 1 1 0])
        @test !_has_dependent_cols(F_1)
        @test !_has_dependent_cols(F_2)
    end

    @testset "_has_dependent_vectors" begin
        @test _has_dependent_vectors([1 0 1; 1 0 1; 0 1 0])
        @test _has_dependent_vectors([1 1 0; 0 0 1; 1 1 0])
        @test !_has_dependent_vectors(F_1)
        @test !_has_dependent_vectors(F_2)
    end

    @testset "_drop_dependent_rows" begin
        M = [1 0 1; 1 0 1; 0 1 0]
        @test _drop_dependent_rows(M) == [1 0 1; 0 1 0]
        M = [1 0 1; -1 0 -1; 0 1 0]
        @test _drop_dependent_rows(M) == [1 0 1; 0 1 0]
        @test _drop_dependent_rows(F_1) == F_1
        @test _drop_dependent_rows(F_2) == F_2
    end

    @testset "_drop_dependent_cols" begin
        M = [1 1 0; 0 0 1; 1 1 0]
        @test _drop_dependent_cols(M) == [1 0; 0 1; 1 0]
        M = [1 -1 0; 0 0 1; 1 -1 0]
        @test _drop_dependent_cols(M) == [1 0; 0 1; 1 0]
        @test _drop_dependent_cols(F_1) == F_1
        @test _drop_dependent_cols(F_2) == F_2
    end

    @testset "_reduce" begin
        # Valid {-1,0,1} matrix reduces correctly
        ok, N = _reduce(F_1)
        @test ok
        @test N == F_1  # F_1 is already fully reduced

        # Matrix with entry outside {-1,0,1} returns false
        ok, _ = _reduce([2 0; 0 1])
        @test !ok

        # Matrix with trivial rows gets reduced
        ok, N = _reduce([1 0 1; 1 0 0; 0 1 1])  # second row is standard basis
        @test ok
        @test !any(i -> count(!iszero, N[i,:]) <= 1, 1:size(N,1))

        # Matrix with dependent rows gets reduced
        ok, N = _reduce([1 0 1; 1 0 1; 0 1 1])
        @test ok
        @test size(N, 1) < 3
    end
    
end
