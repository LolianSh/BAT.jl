# This file is a part of BAT.jl, licensed under the MIT License (MIT).

using BAT
using Compat.Test

@testset "tjelmeland" begin
    @test BAT.multipropT2([135, 288, 64] / 487) ≈ [0, 359, 64] / 423
    @test BAT.multipropT2([0.2, 0.5, 0.3]) ≈ [0.0, 0.625, 0.375]
    @test sum(BAT.multipropT2([123, 12, 324, 3, 31] / 493)) ≈ 1.

    @test_throws ArgumentError BAT.multipropT2([-0.1, 0.1])
    @test_throws ArgumentError BAT.multipropT2([0.1, 0.8])
end
