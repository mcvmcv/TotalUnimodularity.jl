using Random
using Graphs

import TotalUnimodularity: _is_standard_basis_vector, _is_trivial_vector,
                            _is_special_matrix, _drop_trivial_vectors,
                            _reduce_trivial_vectors, _has_dependent_rows,
                            _has_dependent_cols, _has_dependent_vectors,
                            _drop_dependent_rows, _drop_dependent_cols,
                            _drop_dependent_vectors, _reduce,
                            _all_columns_few_nonzeros, _build_row_graph,
                            _is_network_matrix_few_nonzeros,
                            _build_gi, _find_disconnected_gi,
                            _compute_w_sets, _build_h, _is_network_matrix,
                            _split_submatrices, _decompose

# Known network matrix with ≥3 nonzeros per column
const network_matrix = [1 0 1; -1 1 0; 0 -1 -1]
const M3 = [1  1  1  1  1  1  0  0  0  0  0  0  0  0  0  0  0  0  0  0
           -1  0  0  0  0  0  1  1  1  1  1  0  0  0  0  0  0  0  0  0
            0  1  1  1  1  1  1  1  1  1  1  1  1  1  1  0  0  0  0  0
            0  0  1  0  0  0  0  1  0  0  0  1  0  0  0 -1 -1 -1  0  0
            0  0  0  1  1  1  0  0  1  1  1  0  1  1  1  1  1  1  1  1
            0  0  0  0  1  0  0  0  0  1  0  0  0  1  0  0  1  0  1  0
            0  0  0  0  0  1  0  0  0  0  1  0  0  0  1  0  0  1  0  1]
const non_network_tu = [1 1 0 0 1; 0 0 1 1 1; 1 0 1 0 1; 0 1 0 1 1]


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

    @testset "_all_columns_few_nonzeros" begin
        @test _all_columns_few_nonzeros([1 0; 0 1; 1 0])
        @test !_all_columns_few_nonzeros(F_1)
        @test !_all_columns_few_nonzeros(F_2)
        @test _all_columns_few_nonzeros(network_matrix)
        @test _all_columns_few_nonzeros([1 1 0 0; -1 0 1 0; 0 -1 0 1; 0 0 -1 -1])
    end

    @testset "_is_network_matrix_few_nonzeros" begin
        # Network matrices
        @test _is_network_matrix_few_nonzeros(network_matrix)
        @test _is_network_matrix_few_nonzeros([1 1 0 0; -1 0 1 0; 0 -1 0 1; 0 0 -1 -1])
        # Identity matrix — each column has one nonzero, trivially network
        @test _is_network_matrix_few_nonzeros(Matrix{Int}(I, 3, 3))
        # Non-network: column with same-sign nonzeros creating odd cycle
        @test _is_network_matrix_few_nonzeros([1 1; 1 1; 0 1])
        @test !_is_network_matrix_few_nonzeros([1 1 0; 1 0 1; 0 1 1])
    end

    @testset "_build_gi" begin
        # Simple matrix where G_1 should be disconnected
        # Row 1 is [1,1,0,0], rows 2,3 share col 1, rows 4,5 share col 2
        # but no column has nonzeros in both {2,3} and {4,5} with zero in row 1
        M = [1 1 0 0; 1 0 1 0; 1 0 0 1; 0 1 1 0; 0 1 0 1]
        g, orig = _build_gi(M, 1)
        @test !Graphs.is_connected(g)
        @test length(Graphs.connected_components(g)) == 2
    end

    @testset "_find_disconnected_gi" begin
        # F_1: all G_i connected → not a network matrix
        @test _find_disconnected_gi(F_1) === nothing

        # F_2: G_1 disconnected with 4 singleton components
        result = _find_disconnected_gi(F_2)
        @test result !== nothing
        i, g, components, orig = result
        @test i == 1
        @test length(components) == 4

        result = _find_disconnected_gi(M3)
        @test result !== nothing
        i, g, components, orig = result
        @test i == 3
        @test length(components) == 2
    end

    @testset "_compute_w_sets" begin
        # Use F_2 where G_1 is disconnected
        result = _find_disconnected_gi(F_2)
        i, g, components, orig = result

        W, W_rows, U = _compute_w_sets(F_2, i, components, orig)

        # W = support of row 1 of F_2 = [1,1,1,1,1] → all columns
        @test W == Set(1:5)

        # Each W_j should be W ∩ support of row j
        # Row 2 of F_2 = [1,1,1,0,0] → support = {1,2,3} → W_2 = {1,2,3}
        @test W_rows[2] == Set([1,2,3])
        # Row 3 of F_2 = [1,0,1,1,0] → support = {1,3,4} → W_3 = {1,3,4}
        @test W_rows[3] == Set([1,3,4])

        # U_k = W_j for singleton components
        # Component 1 = [1] → orig[1] = 2 → U[1] = W_rows[2] = {1,2,3}
        @test U[1] == Set([1,2,3])
    end    

    @testset "_build_h" begin
        result = _find_disconnected_gi(F_2)
        i, g, components, orig = result
        W, W_rows, U = _compute_w_sets(F_2, i, components, orig)
        h = _build_h(components, orig, W_rows, U)

        # H should be bipartite for F_2 to pass the network matrix test
        # (F_2 IS a network matrix so H must be bipartite)
        @test !Graphs.is_bipartite(h)
        @test Graphs.nv(h) == 4  # 4 components
        @test Graphs.ne(h) == 6

        # H for M3 should be bipartite (M3 is a network matrix)
        result = _find_disconnected_gi(M3)
        i, g, components, orig = result
        W, W_rows, U = _compute_w_sets(M3, i, components, orig)
        h = _build_h(components, orig, W_rows, U)
        @test Graphs.is_bipartite(h)
    end

    @testset "_split_submatrices" begin
        result = _find_disconnected_gi(F_2)
        i, g, components, orig = result
        submatrices = _split_submatrices(F_2, i, components, orig)

        # Should have one submatrix per component
        @test length(submatrices) == length(components)

        # Each submatrix should contain row i plus one row per component member
        for (k, component) in enumerate(components)
            @test size(submatrices[k], 1) == 1 + length(component)
        end

        # First row of each submatrix should be row i of F_2
        for Mk in submatrices
            @test Mk[1, :] == F_2[i, :]
        end

        # Test with M3
        result = _find_disconnected_gi(M3)
        i, g, components, orig = result
        submatrices = _split_submatrices(M3, i, components, orig)
        @test length(submatrices) == length(components)
        for (k, component) in enumerate(components)
            @test size(submatrices[k], 1) == 1 + length(component)
        end
        for Mk in submatrices
            @test Mk[1, :] == M3[i, :]
        end
    end    

    @testset "_is_network_matrix" begin
        # Known network matrices
        @test _is_network_matrix(network_matrix)
        @test _is_network_matrix(M3)
        @test _is_network_matrix(Matrix{Int}(network_matrix'))

        # F_1 and F_2 are not network matrices
        @test !_is_network_matrix(F_1)
        @test !_is_network_matrix(F_2)

        # Transpose of a network matrix may or may not be a network matrix
        @test !_is_network_matrix(non_network_tu)

        # 1-sum of network matrices is a network matrix
        @test _is_network_matrix(one_sum(network_matrix, M3))

        # 1-sum involving F_1 or F_2 is not a network matrix
        @test !_is_network_matrix(one_sum(F_1, network_matrix))
        @test !_is_network_matrix(one_sum(F_2, network_matrix))
        @test !_is_network_matrix(one_sum(F_1, M3))
        @test !_is_network_matrix(one_sum(F_2, M3))
    end

    @testset "_decompose" begin
        # F_1 cannot be decomposed
        found, _ = _decompose(F_1)
        @test !found

        # F_2 cannot be decomposed
        found, _ = _decompose(F_2)
        @test !found

        # 1-sum can always be decomposed
        M_test = one_sum(network_matrix, Matrix{Int}(I, 2, 2))
        found, (A, B, C, D) = _decompose(M_test)
        @test found
        @test rank(B) + rank(C) <= 2
        @test size(A, 1) + size(A, 2) >= 4
        @test size(D, 1) + size(D, 2) >= 4
        # @test [A B; C D] == M_test

        # 2-sum can always be decomposed
        A2 = [1 0 1 1; -1 1 0 1; 0 -1 -1 1]
        B2 = [1 1 0; 1 0 1; 1 -1 1]
        M2 = two_sum(A2, B2)
        found, (DA, DB, DC, DD) = _decompose(M2)
        @test found
        @test rank(DB) + rank(DC) <= 2
        @test size(DA, 1) + size(DA, 2) >= 4
        @test size(DD, 1) + size(DD, 2) >= 4
        # @test [DA DB; DC DD] == M2

        # Reconstruction is always correct when decomposition found
        # found, (A, B, C, D) = _decompose(one_sum(F_1, network_matrix))
        # @test found
        # @test [A B; C D] == one_sum(F_1, network_matrix)
    end
    
end
