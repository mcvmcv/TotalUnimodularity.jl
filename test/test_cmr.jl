# Tests ported from CMR's test_tu.cpp (cmr-main/test/test_tu.cpp).
# Each @testset corresponds to a TEST(TU, ...) case in the CMR suite.
# Matrices are taken verbatim from the CMR source; composition tests use the
# Julia two_sum / one_sum functions (which may produce a different row/column
# permutation than CMR's compose functions, but TU is invariant under those).

using TotalUnimodularity
using Test

# K_{3,3} representation matrix (5×4) — appears in several CMR tests.
const K33 = [
  1  1  0  0
  1  1  1  0
  1  0  0 -1
  0  1  1  1
  0  0  1  1
]

# K_{3,3} dual representation matrix used in TwoSum tests (4×5).
const K33dual_twosum = [
  1  1  1  0  0
  1  1  0  1  0
  0  1  0  1  1
  0  0 -1  1  1
]

# K_{3,3} dual representation matrix used in the Onesum test (4×5, different signs).
const K33dual_onesum = [
  1  1  1  0  0
  1  1  0 -1  0
  0  1  0 -1 -1
  0  0  1  1  1
]

@testset "CMR TU tests" begin

  @testset "EulerianAlgorithm" begin
    # Sub-case 1: 2-sum of K_{3,3} and its dual → TU.
    @test is_totally_unimodular(two_sum(K33, K33dual_twosum))

    # Sub-case 2: 12×12 non-TU matrix.
    M = [
      1 1 1 0 1 0 1 1 1 1 1 1
      1 0 1 0 1 0 1 1 1 1 1 0
      0 1 1 0 0 0 0 0 0 0 0 0
      0 1 1 1 0 0 0 0 0 0 0 0
      0 1 1 1 1 0 0 0 0 0 0 0
      0 1 1 1 1 1 0 0 0 0 0 0
      0 1 1 1 1 1 1 0 0 0 0 0
      0 0 0 0 0 0 1 1 0 0 0 0
      0 0 0 0 0 0 0 1 1 0 0 0
      0 0 0 0 0 0 0 0 1 1 0 0
      0 0 0 0 0 0 0 0 0 1 1 0
      0 0 0 0 0 0 0 0 0 0 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "PartitionAlgorithm" begin
    # Sub-case 1: 2-sum of K_{3,3} and its dual → TU.
    @test is_totally_unimodular(two_sum(K33, K33dual_twosum))

    # Sub-case 2: 14×14 non-TU matrix.
    M = [
      1 1 1 0 1 0 1 0  1 1 1 1 1 1
      1 0 1 0 1 0 1 0  1 1 1 1 1 0
      0 1 1 0 0 0 0 0  0 0 0 0 0 0
      0 1 1 1 0 0 0 0  0 0 0 0 0 0
      0 1 1 1 1 0 0 0  0 0 0 0 0 0
      0 1 1 1 1 1 0 0  0 0 0 0 0 0
      0 1 1 1 1 1 1 0  0 0 0 0 0 0
      0 1 1 1 1 1 1 1  0 0 0 0 0 0
      0 1 1 1 1 1 1 1  1 0 0 0 0 0
      0 0 0 0 0 0 0 0  1 1 0 0 0 0
      0 0 0 0 0 0 0 0  0 1 1 0 0 0
      0 0 0 0 0 0 0 0  0 0 1 1 0 0
      0 0 0 0 0 0 0 0  0 0 0 1 1 0
      0 0 0 0 0 0 0 0  0 0 0 0 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "Onesum" begin
    # 1-sum (block diagonal) of K_{3,3} and its dual → TU.
    @test is_totally_unimodular(one_sum(K33, K33dual_onesum))
  end

  @testset "SeriesParallelTwoSeparation" begin
    @test is_totally_unimodular(two_sum(K33, K33dual_twosum))
  end

  @testset "NestedMinorSearchTwoSeparation" begin
    @test is_totally_unimodular(two_sum(K33, K33dual_twosum))
  end

  @testset "NestedMinorSearchTwoSeparationViolator" begin
    M = [
       1  1  0  0  0  0  0  0
       1  0  0 -1  0  0  0  0
       0  1  1  1  0  0  0  0
       0  0  1  1  0  0  0  0
       1  1  1  0  1  1  0  0
       1  1  1  0  1  0  1  0
       1  1 -1  0  0  0  1  1
       0  0  0  0  0 -1  1  1
    ]
    @test !is_totally_unimodular(M)
  end

  # The NestedMinorPivots tests exercise specific decomposition paths in CMR.
  # The CMR source (test_tu.cpp) does not assert TU for these matrices.
  # Naive verification shows they are NOT TU (each has a 3×3 submatrix with
  # det=2), so no TU assertion is made here.
  @testset "NestedMinorPivotsOneRowOneColumn" begin
    M = [
      1 0 1 0 0 0
      1 1 0 0 0 1
      0 1 1 0 0 0
      0 0 0 0 1 1
      0 0 0 1 1 0
      0 1 1 1 0 0
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "NestedMinorPivotsTwoRowsOneColumn" begin
    M = [
      1 0 1 0 0 0
      1 1 0 0 0 0
      0 1 1 0 0 0
      0 0 1 0 0 1
      0 0 0 0 1 1
      0 0 0 1 1 0
      0 1 1 1 0 0
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "NestedMinorPivotsOneRowTwoColumns" begin
    M = [
      1 0 1 0 0 0 0
      1 1 0 0 0 0 1
      0 1 1 1 0 0 1
      0 0 0 0 0 1 1
      0 0 0 0 1 1 0
      0 0 0 1 1 0 0
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "NestedMinorPivotsTwoSeparation" begin
    M = [
      1 0 1 0 0 0 0 0 0 0 0
      1 1 0 0 0 1 0 0 0 0 0
      0 1 1 0 0 0 0 0 0 0 0
      0 0 0 0 1 1 0 0 0 0 0
      0 0 0 1 1 0 0 0 0 0 1
      0 1 1 1 0 0 0 0 0 1 0
      0 0 0 0 0 0 0 1 1 0 0
      0 0 0 0 0 0 1 1 0 0 0
      0 1 1 0 0 0 1 0 0 0 0
      0 1 1 0 0 0 0 0 1 0 0
      0 0 0 0 0 0 0 0 0 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "SeqGraphicWheel" begin
    # 6×6 wheel (graphic → TU)
    M1 = [
      1 0 0 0 0 1
      1 1 0 0 0 0
      0 1 1 0 0 0
      0 0 1 1 0 0
      0 0 0 1 1 0
      0 0 0 0 1 1
    ]
    @test is_totally_unimodular(M1)

    # Four 4×4 graphic matrices
    M2 = [
      1  0  1  0
      1 -1  0  0
      0  1  1  1
      0  0  1  1
    ]
    @test is_totally_unimodular(M2)

    M3 = [
      1 1 1 0
      1 1 0 0
      0 1 1 1
      0 0 1 1
    ]
    @test is_totally_unimodular(M3)

    M4 = [
      1 0 1 0
      1 1 1 0
      0 1 1 1
      0 0 1 1
    ]
    @test is_totally_unimodular(M4)

    M5 = [
      1 0 1 0
      1 1 0 0
      1 1 1 1
      0 0 1 1
    ]
    @test is_totally_unimodular(M5)
  end

  @testset "SeqGraphicOneRowOneColumn" begin
    M = [
      1 0 0 1 0 0 0
      1 1 0 0 0 0 0
      0 1 1 0 0 0 0
      0 0 1 1 1 1 1
      0 0 0 1 1 1 1
      0 0 1 1 1 0 0
      0 0 0 0 1 0 1
    ]
    @test is_totally_unimodular(M)
  end

  @testset "SeqGraphicTwoRowsOneColumn" begin
    M = [
      1  0  0  1  0  0
      1  1  0  0  0  0
      0  1  1  0  0  0
      0  0  1  1  0  0
      0  0  0  1  1  0
      0  0  1  1  1  0
      0  0  0  1  0  1
      0  0  0  0 -1  1
    ]
    @test is_totally_unimodular(M)
  end

  @testset "SeqGraphicOneRowTwoColumns" begin
    M = [
      1  0  0  1  0  0  0  0
      1  1  0  0  0  0  0  0
      0  1  1  0 -1  0 -1  0
      0  0  1  1  0  1  0  1
      0  0  0  0  1  1  1  1
      0  0  0  0  0  0  1  1
    ]
    @test is_totally_unimodular(M)
  end

  @testset "SeqGraphicnOneColumn" begin
    M = [
      1 0 0 1 0 0
      1 1 0 0 0 0
      0 1 1 0 0 0
      0 0 1 1 0 1
      0 0 0 1 1 0
      0 0 1 1 1 1
    ]
    @test is_totally_unimodular(M)
  end

  @testset "SeqGraphicOneRow" begin
    # Sub-case 1: 6×5 graphic → TU
    M1 = [
      1  1  0  0  0
      0  1  1  0  1
      1  0 -1  1  0
      1  0  0  1  0
      0  1  0  0  1
      0  1  1  0  0
    ]
    @test is_totally_unimodular(M1)

    # Sub-case 5: 5×5 graphic → TU
    M5 = [
      1  1  0  0  1
      1  1  1  0  1
      0  1  1  1  0
      1  0  0 -1  0
      0  1  1  1  1
    ]
    @test is_totally_unimodular(M5)

    # Sub-cases 2–4: the CMR test checks graphicness only (not TU directly).
    # We include them here as known non-graphic but potentially-TU matrices;
    # the CMR test does not assert isTU for these, so we skip TU assertions.
  end

  @testset "R10" begin
    # 5×5 R10 matrix — TU (regular, neither graphic nor cographic)
    M1 = [
      1 0 0 1 1
      1 1 0 0 1
      0 1 1 0 1
      0 0 1 1 1
      1 1 1 1 1
    ]
    @test is_totally_unimodular(M1)

    # Alternative signing of R10
    M2 = [
      1  1  0  0  1
      1  1 -1  0  0
      0  1 -1 -1  0
      0  0  1  1  1
      1  0  0  1  1
    ]
    @test is_totally_unimodular(M2)
  end

  @testset "EnumerateRanksZeroTwo" begin
    M = [
      1 0 1 1
      1 1 1 0
      0 0 1 1
      0 1 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "EnumerateRanksOneOne" begin
    M = [
      1 1 1 1
      0 1 1 1
      1 0 0 1
      1 1 0 0
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "EnumerateRanksTwoZero" begin
    M = [
      1 1 0 1 0
      1 1 1 0 1
      1 1 1 0 0
      1 1 1 1 1
      1 0 0 0 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "R12 (DeltasumR12 / YsumR12 / ThreesumR12)" begin
    # The same 6×6 R12 matrix is used in all three tests → TU
    M = [
      1  0  1  1  0  0
      0  1  1  1  0  0
      1  0  1  0  1  1
      0 -1  0 -1  1  1
      1  0  1  0  1  0
      0 -1  0 -1  0  1
    ]
    @test is_totally_unimodular(M)
  end

  @testset "ThreesumForbiddenSubmatrix" begin
    # 14×14 — same matrix reused in Ysum and Deltasum forbidden-submatrix tests
    M = [
      1 1 1 0 1 0 1 0 1 1 1 1 1 1
      1 0 1 0 1 0 1 0 1 1 1 1 1 0
      0 1 1 0 0 0 0 0 0 0 0 0 0 0
      0 1 1 1 0 0 0 0 0 0 0 0 0 0
      0 1 1 1 1 0 0 0 0 0 0 0 0 0
      0 1 1 1 1 1 0 0 0 0 0 0 0 0
      0 1 1 1 1 1 1 0 0 0 0 0 0 0
      0 1 1 1 1 1 1 1 0 0 0 0 0 0
      0 1 1 1 1 1 1 1 1 0 0 0 0 0
      0 0 0 0 0 0 0 0 1 1 0 0 0 0
      0 0 0 0 0 0 0 0 0 1 1 0 0 0
      0 0 0 0 0 0 0 0 0 0 1 1 0 0
      0 0 0 0 0 0 0 0 0 0 0 1 1 0
      0 0 0 0 0 0 0 0 0 0 0 0 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "Fano" begin
    # 3×4 Fano matroid representation — not TU
    M = [
      1 1 0 1
      0 1 1 1
      1 0 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "FanoDual" begin
    # 4×3 Fano dual representation — not TU
    M = [
      1 1 0
      0 1 1
      1 0 1
      1 1 1
    ]
    @test !is_totally_unimodular(M)
  end

  @testset "ThreesumPivotHighRank" begin
    # 7×7 — not TU
    M1 = [
      1  1  1  0  0  0  0
      0  0  1  1  1  1  0
      0  1  0  0  0 -1  1
      0  1  0  0  0  0  1
      0  1  0 -1 -1  0  0
      1  1  0 -1  0  0  1
      0  0  0  1  1  0  0
    ]
    @test !is_totally_unimodular(M1)

    # 10×10 — not TU
    M2 = [
       0  0  0  0  0  0  0  0  1  0
       0  0  1  0  0  0  0  0  0 -1
       0  1  0  0  0  0  0  0  0  0
       0 -1  1  1  0  0  0  0  0  0
       0  1  0  0  0  1  1  1  1  1
       1  0  1  1  0  0  0  1  1  0
       1  0  0  0  0  0  0  1  0  0
       0  1  0  0  0  0  0  0  0  0
       0 -1  0  1  0  0  0  1  0  0
       0  0  0  1  1 -1  0  1  0  0
    ]
    @test !is_totally_unimodular(M2)
  end

  @testset "CompleteTree" begin
    # All sub-cases are 1-sums of two irregular blocks → not TU.

    M1 = [
      1  1  0  1  0  0  0  0
      0  1  1  1  0  0  0  0
      1  0 -1  1  0  0  0  0
      1  1  1  0  0  0  0  0
      0  0  0  0  1  0  1  1
      0  0  0  0  1  1  0  1
      0  0  0  0  0  1 -1  1
      0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M1)

    M2 = [
      1  1  0  1  0  0  0  0
      0  1  1  0  0  0  0  0
      1  0  1  0  0  0  0  0
      1  0  0  0  0  0  0  0
      0  0  0  0  1  0  1  1
      0  0  0  0  1  1  0  1
      0  0  0  0  0  1 -1  1
      0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M2)

    M3 = [
      1  1  0  0  0  0  0  0
      0  1  1  0  0  0  0  0
      0  0 -1  1  0  0  0  0
      1  0  0  1  0  0  0  0
      0  0  0  0  1  0  1  1
      0  0  0  0  1  1  0  1
      0  0  0  0  0  1 -1  1
      0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M3)

    M4 = [
      1  0  1  1  0  0  0  0
      1  1  0  0  0  0  0  0
      0  1 -1  1  0  0  0  0
      1  1  1  1  0  0  0  0
      0  0  0  0  1  0  1  1
      0  0  0  0  1  1  0  1
      0  0  0  0  0  1 -1  1
      0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M4)

    M6 = [
      1  1  0  0  0  0  0  0  0  0
      1  0  0 -1  0  0  0  0  0  0
      0  1  1  1  0  0  0  0  0  0
      0  0  1  1  0  0  0  0  0  0
      1  1  1  0  1  0  0  0  0  0
      1  1  1  0  1  1  0  0  0  0
      1  1 -1  0  0  1  0  0  0  0
      0  0  0  0  0  0  0 -1  1  1
      0  0  0  0  0  0  1  0  1  1
      0  0  0  0  0  0  1  1  0  1
      0  0  0  0  0  0  0  1 -1  1
      0  0  0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M6)

    M7 = [
      -1  0  0  1  1  0  0  0  0
       1  1  0  0  1  0  0  0  0
       0  1  1  0  1  0  0  0  0
       0  0  1  1  1  0  0  0  0
       1  1  1  1  1  0  0  0  0
       0  0  0  0  0  1  0  1  1
       0  0  0  0  0  1  1  0  1
       0  0  0  0  0  0  1 -1  1
       0  0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M7)

    M8 = [
      1  1  0  1  0  0  0  0
      0  1  1  0  0  0  0  0
      1  0  1  0  0  0  0  0
      1  0  0  0  0  0  0  0
      0  0  0  0  1  0  1  1
      0  0  0  0  1  1  0  1
      0  0  0  0  0  1  1  1
      0  0  0  0  1  1  1  0
    ]
    @test !is_totally_unimodular(M8)
  end

end
