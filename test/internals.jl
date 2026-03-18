import TotalUnimodularity: _is_standard_basis_vector, _is_trivial_vector,
                            _is_special_matrix, _drop_trivial_vectors,
                            _reduce_trivial_vectors

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

end
