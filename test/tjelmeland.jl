# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "tjelmeland" begin
    @test BAT.multipropT2([135, 288, 64] / 487) ≈ [0, 359, 64] / 423
    @test BAT.multipropT2([0.2, 0.5, 0.3]) ≈ [0.0, 0.625, 0.375]
    @test sum(BAT.multipropT2([123, 12, 324, 3, 31] / 493)) ≈ 1.


    pdist = GenericProposalDist(MvNormal([0., 1.], [1. .5; .5 2.]))
    target = MvNormal([2., 1.], [1. .6; .6 8.])
    rng = MersenneTwister(8477)
    params_old = [1. , 2.]
    P_T1 = BAT.multipropT1(rng, pdist, target, params_old, 100)
    @test sum(P_T1) ≈ 1.
    @test !any(x -> x < 0, P_T1)

    pdist = GenericProposalDist(MvNormal([0.], [1.]))
    target = MvNormal([2.4], [.5])
    rng = MersenneTwister(8477)
    params_old = [1.]
    P_T1 = BAT.multipropT1(rng, pdist, target, params_old, 100)
    @test sum(P_T1) ≈ 1.
    @test !any(x -> x < 0, P_T1)

    @test_throws ArgumentError BAT.multipropT2([-0.1, 0.1])
    @test_throws ArgumentError BAT.multipropT2([0.1, 0.8])
end
