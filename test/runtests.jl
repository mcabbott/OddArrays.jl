using Test, OddArrays, LinearAlgebra
using Zygote, ForwardDiff

@testset "rotations" begin
    @test Rotation(pi/2) ≈ [0 -1; 1 0]
    @test _collect(Rotation(pi/2)) ≈ [0 -1; 1 0]

    @test_broken Rotation(pi/4) * [1, 0] ≈ [sqrt(2), sqrt(2)]
    @test _collect(Rotation(pi/4)) * [1, 0] ≈ [sqrt(2)/2, sqrt(2)/2]

    @test decollect([1 2; 3 4], Rotation(pi/7)).theta ≈ _decollect([1 2; 3 4], Rotation(pi/7)).theta

    @test gradient(x -> _getindex(x,1,2), Rotation(pi/7))[1].theta ≈ gradient(x -> x[1,2], Rotation(pi/7))[1].theta
end

# These are all different. So you need the primal to convert back.
# But at the Tangent + Matrix stage, you don't have it,
# which is why I think you have to do this soon, ProjectTo
_decollect([0 1; 0 0], Rotation(pi/2))
_decollect([0 1; 0 0], Rotation(pi/3))
_decollect([0 1; 0 0], Rotation(pi/4))

decollect([0 0 1; 0 0 0; 0 0 0], Vandermonde([2,3,4]))
decollect([0 0 1; 0 0 0; 0 0 0], Vandermonde([5,6,7]))

@testset "logrange" begin
    @test LogRange(1,4,3) ≈ [1,2,4]
    @test _collect(LogRange(1,4,3)) ≈ [1,2,4]
    @test _collect(LogRange(3,11,13)) ≈ LogRange(3,11,13)

    @test _prod(LogRange(3,11,13)) ≈ prod(Vector(LogRange(3,11,13)))
end

@testset "vandermonde" begin
    @test Vandermonde([3,4,5]) ≈ hcat(([3,4,5] .^ i for i in 0:2)...)
    @test Vandermonde([3,4,5,6],7) ≈ hcat(([3,4,5,6] .^ i for i in 0:3)...) .* 7
    @test Vandermonde([3,4,5]) ≈ _collect(Vandermonde([3,4,5]))

    x1 = Vandermonde([1.2, 3.4, 5.6, 7.8])
    @test _det(x1) ≈ det(Matrix(x1))

    x2 = Vandermonde([1.2, 3.4, 5.6], 7.8)
    @test _det(x2) ≈ det(Matrix(x2))

    @test gradient(v -> _det(Vandermonde(v)), [2,3,4])[1] ≈ ForwardDiff.gradient(v -> det(Matrix(Vandermonde(v))), [2,3,4])
    v0 = [1.2, 3.4, 5.6, 7.8]
    @test gradient(s -> _det(Vandermonde(v0, s)), 9.1)[1] ≈ ForwardDiff.derivative(s -> det(Matrix(Vandermonde(v0, s))), 9.1)
end

@testset "outer" begin
    @test Outer([1,2], [3,4]) == [3 4; 6 8]
    @test Outer(10, [3 4; 5 6]) == [30 50; 40 60]
    @test Outer([3 4; 5 6], 10) == [30 40; 50 60]
    @test Outer([1 2; 3 4], [5 6; 7 8]) == [1 2; 3 4] * [5 6; 7 8]'

    @test decollect([1 10; 100 1000], Outer([3,4], [5,6])) == _decollect([1 10; 100 1000], Outer([3,4], [5,6]))
    @test decollect([1 10; 100 1000], Outer([3 4; 5 6], 7)) == _decollect([1 10; 100 1000], Outer([3 4; 5 6], 7))
    @test decollect([1 10; 100 1000], Outer(2, [3 4; 5 6])) == _decollect([1 10; 100 1000], Outer(2, [3 4; 5 6]))
end

@testset "full" begin
    @test Full(pi,2,3) == fill(pi,2,3)

    @test Full(pi,2,3) * Full(ℯ,3) ≈ fill(pi,2,3) * fill(ℯ,3)
    @test Full(pi,2,3) * Full(ℯ,3,4) ≈ fill(pi,2,3) * fill(ℯ,3,4)

    @test decollect([1,10,100], Full(2,3)) == _decollect([1,10,100], Full(2,3))

    @test gradient(x -> sum(Full(x,2,3) * Full(x,3,4)), 5) == gradient(x -> sum(fill(x,2,3) * fill(x,3,4)), 5)
end

nothing_to_zero(::Nothing) = 0
nothing_to_zero(x) = x
nothing_to_zero(xs::AbstractArray) = map(nothing_to_zero, xs)

@testset "mask" begin
    @test Mask([1,Inf,3,nothing,5], [10,20,30,40,50]) == [1,20,3,40,5]

    d1 = decollect([10,100], Mask([3,missing], [4,5]))
    d2 = _decollect([10,100], Mask([3,missing], [4,5]))
    @test d1 == map(nothing_to_zero, d2)
end
