using Test, OddArrays, LinearAlgebra
using Zygote, ForwardDiff, ChainRulesCore

# These are all different. So you need the primal to convert back.
# But at the Tangent + Matrix stage, you don't have it,
# which is why I think you have to do this soon, ProjectTo
_uncollect([0 1; 0 0], Rotation(pi/2))
_uncollect([0 1; 0 0], Rotation(pi/3))
_uncollect([0 1; 0 0], Rotation(pi/4))

uncollect([0 0 1; 0 0 0; 0 0 0], Vandermonde([2,3,4]))
uncollect([0 0 1; 0 0 0; 0 0 0], Vandermonde([5,6,7]))

@testset "rotations" begin
    @test Rotation(pi/2) ≈ [0 -1; 1 0]
    @test _collect(Rotation(pi/2)) ≈ [0 -1; 1 0]

    @test Rotation(pi/4) * [1, 0] ≈ [sqrt(2)/2, sqrt(2)/2]
    @test _collect(Rotation(pi/4)) * [1, 0] ≈ [sqrt(2)/2, sqrt(2)/2]

    @test uncollect([1 2; 3 4], Rotation(pi/7)).theta ≈ _uncollect([1 2; 3 4], Rotation(pi/7)).theta

    @test gradient(x -> _getindex(x,1,2), Rotation(pi/7))[1].theta ≈ gradient(x -> x[1,2], Rotation(pi/7))[1].theta

    # restrict/naturalise axioms, new
    @test uncollect(collect(restrict([3 4; 5 6], Rotation(7))), Rotation(7)) ≈ uncollect([3 4; 5 6], Rotation(7))
    if uncollect([3 4; 0 0], Rotation(5)) ≈ uncollect([0 0; -4 3], Rotation(5))  # very week, == indep theta!
        @test restrict([3 4; 0 0], Rotation(5)) ≈ restrict([0 0; -4 3], Rotation(5))
    end
    @test naturalise(uncollect([3 4; 5 6], Rotation(2)), Rotation(2)) ≈ restrict([3 4; 5 6], Rotation(2))

    # adjoint constructor? Nothing special needed
    @test gradient(x -> sum(Rotation(x)[:,1]), 1)[1] ≈ ForwardDiff.derivative(x -> sum(Rotation(x)[:,1]), 1)

    # generic functions: now test with _mul instead of opting out
    @test_throws Exception gradient(x -> (x * x)[1,2], Rotation(pi/7))[1]
    @test_throws Exception gradient(x -> _getindex(x * x, 1,2), Rotation(pi/7))[1]
    # ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::Rotation, ::Rotation)
    # ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(inv), ::Rotation)
    @test gradient(x -> _getindex(_mul(x, x), 1,2), Rotation(pi/7))[1].theta ≈ ForwardDiff.derivative(x -> (Rotation(x)*Rotation(x))[1,2], pi/7)
    @test gradient(x -> (_mul(x, _inv(x)))[1,2], Rotation(pi/7))[1] ≈ (; theta=0)
    # @test_broken gradient(x -> (x*_inv(x))[1,2], Rotation(pi/7))[1] ≈ (; theta=0)  # inv weirdly fails to opt-out?
    @test gradient(x -> (_mul(x, _inv(x)))[1,2], Rotation(pi/7))[1] ≈ (; theta=0)
end

@testset "rotations + antisymm" begin
    @test AntiSymOne(3//4) == [0 -0.75; 0.75 0]
    @test exp(AntiSymOne(3//4)) ≈ exp(collect(float(AntiSymOne(3//4)))) ≈ Rotation(3//4)
    @test exp(AntiSymOne(3//4)) isa Rotation

    @test AntiSymOne(1.23) * AntiSymOne(4.56) ≈ collect(AntiSymOne(1.23)) * collect(AntiSymOne(4.56))
    @test AntiSymOne(-1) * Rotation(pi/3) ≈ Rotation(-pi/6)  # but general case will scale the Rotation

    # Now hook that into gradients?
    ChainRulesCore.ProjectTo(r::Rotation) = ProjectTo{Rotation}(; x = r)
    function (p::ProjectTo{Rotation})(dx::AbstractArray)
        @info "projecting to AntiSymOne!" typeof(dx)
        restrict(dx, p.x)
    end
    function (p::ProjectTo{Rotation})(dx::Tangent)
        @info "projecting Tangent -> AntiSymOne"
        naturalise(dx, p.x)
    end
    @test gradient(x -> (x * x)[1,2], Rotation(pi/7))[1] isa AntiSymOne
    @test_skip gradient(x -> (Rotation(x) * Rotation(x))[1,2], pi/7)[1]  # needs an adjoint constructor
end

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

    @test uncollect([1 10; 100 1000], Outer([3,4], [5,6])) == _uncollect([1 10; 100 1000], Outer([3,4], [5,6]))
    @test uncollect([1 10; 100 1000], Outer([3 4; 5 6], 7)) == _uncollect([1 10; 100 1000], Outer([3 4; 5 6], 7))
    @test uncollect([1 10; 100 1000], Outer(2, [3 4; 5 6])) == _uncollect([1 10; 100 1000], Outer(2, [3 4; 5 6]))
end

nothing_to_zero(::Nothing) = 0
nothing_to_zero(x) = x
nothing_to_zero(xs::AbstractArray) = map(nothing_to_zero, xs)

@testset "mask" begin
    @test Mask([1,Inf,3,nothing,5], [10,20,30,40,50]) == [1,20,3,40,5]

    d1 = uncollect([10,100], Mask([3,missing], [4,5]))
    d2 = _uncollect([10,100], Mask([3,missing], [4,5]))
    @test d1 == map(nothing_to_zero, d2)
end

@testset "pdmat" begin
    @test PDiagMat([1,2,3]) == diagm([1,2,3])
    @test inv(PDiagMat([1,20,300])) ≈ inv(diagm([1,20,300]))

    p2 = PDiagMat([1,2,3]) * PDiagMat([4,5,6])
    @test p2 isa PDiagMat
    @test p2.inv_diag ≈ inv.(p2.diag)

    @test gradient(x -> x[5], PDiagMat([1,2,3]))[1] ≈ (dim=nothing, diag=[0, 1, 0], inv_diag=nothing)

    # adjoint constructor
    @test_skip gradient(x -> sum(PDiagMat(x)), [1,2,3])[1] == [1,1,1]
end

@testset "full, one=$testone" for testone in (false, true)
    ONEGRAD[] = testone

    @test Full(pi,2,3) == fill(pi,2,3)

    @test Full(pi,2,3) * Full(ℯ,3) ≈ fill(pi,2,3) * fill(ℯ,3)
    @test Full(pi,2,3) * Full(ℯ,3,4) ≈ fill(pi,2,3) * fill(ℯ,3,4)

    # correctness of uncollect
    @test uncollect([1,10,100], Full(2,3)) == _uncollect([1,10,100], Full(2,3))

    # naturalise axioms, old
    @test restrict(restrict([0,pi,10], Full), Full) ≈ restrict([0,pi,10], Full) # projection
    @test uncollect(restrict([1,10,100], Full(2,3)), Full(2,3)) == uncollect([1,10,100], Full(2,3))
    @test restrict([1,1,2,3] + [5,8,13,21], Full(pi,4)) ≈ restrict([1,1,2,3], Full(pi,4)) + restrict([5,8,13,21], Full(pi,4))
    if uncollect([1,10,100], Full) == uncollect([100,10,1], Full)
        @test restrict([1,10,100], Full) ≈ restrict([1,10,100], Full) 
    end
    @test uncollect(naturalise((value=12.34, size=nothing), Full(0,3,4)), Full) ≈ (value=12.34, size=nothing)

    # restrict/naturalise axioms, newer
    @test uncollect(collect(restrict([0,pi,10], Full)), Full) ≈ uncollect([0,pi,10], Full)
    if uncollect([1,10,100], Full) == uncollect([100,10,1], Full)
        @test restrict([1,10,100], Full) ≈ restrict([1,10,100], Full)
    end
    @test naturalise(uncollect([0,3,7], Full), Full(0,3)) ≈ restrict([0,3,7], Full)

    # generic functions
    @test gradient(x -> sum(Full(x,2,3) * Full(x,3,4)), 5) == gradient(x -> sum(fill(x,2,3) * fill(x,3,4)), 5)

    # adjoint constructor
    @test gradient(x -> Full(x,3)[2], 1) == (1.0,)
    @test gradient(x -> Full(x,3,3)[4], 5) == (1.0,)
end

@testset "alphabet" begin
    a = Alphabet(true,2,3.0)
    @test a == [1,2,3]
    @test a[1] === 1.0
    @test a.a === 1.0
    a[2] = 99
    @test a[2] === 99.0

    
end

@testset "range" begin
    @test Range(1,pi,7) ≈ range(1,pi,length=7)

    # correctness of uncollect
    @test uncollect([4,-5,9], Range(1,2,3)) ≈ _uncollect([4,-5,9], Range(1,2,3))
    @test uncollect([1,10,100,100], Range(1,pi,4)) ≈ _uncollect([1,10,100,100], Range(1,pi,4))

    # naturalise axioms
    # @test restrict(restrict([0,pi,10], Range), Range) ≈ restrict([0,pi,10], Range) # projection
    # for dx in ([1,2,3], [0,3,7,8], [4,-5,9])
    #     @show dx
    #     r0 = Range(0,0,length(dx))
    #     @test uncollect(restrict(dx, r0), r0) ≈ uncollect(dx, r0)
    # end
    # @test restrict([1,1,2,3] + [5,8,13,21], Range(1,2,4)) ≈ restrict([1,1,2,3], Range(1,2,4)) + restrict([5,8,13,21], Range(1,2,4))  # linear
    # if uncollect([1,10,2], Range) ≈ uncollect([2,8,3], Range)
    #     @test restrict([1,10,2], Range) ≈ restrict([2,8,3], Range)

    #     # @test restrict([1,10,2], Range) ≈ naturalise(uncollect([1,10,2], Range), Range(0,0,3))
    #     # @test restrict([1,10,2], Range) ≈ naturalise(uncollect([2,8,3], Range), Range(0,0,3))
    # end
    # for dx in ([1,2,3], [0,3,7,8],)
    #     # @test naturalise(uncollect(dx, Range), Range(0,0,length(dx))) ≈ restrict(dx, Range)
    # end

    # adjoint constructor
    # gradient(x -> Range(1,x,5)[end], 2) # == (1,)
    # gradient(x -> collect(Range(1,x,5))[end], 2) == (1,)
    # gradient(x -> Range(1,x,5)[1], 2) # == (0,)
    # gradient(x -> collect(Range(1,x,5))[1], 2) == (0,)
end

@testset "Diagonal" begin
    @test _uncollect([1 2 3; 4 5 6; 7 8 9], Diagonal([0,0,0])) == (diag = [1.0, 5.0, 9.0],)
    @test uncollect([1 2 3; 4 5 6; 7 8 9], Diagonal([0,0,0])) == (diag = [1.0, 5.0, 9.0],)

    @test restrict([1 2 3; 4 5 6; 7 8 9], Diagonal) isa Diagonal
    @test restrict([1 2 3; 4 5 6; 7 8 9], Diagonal([0,0,0])) == Diagonal([1,5,9])
end

@testset "Symmetric" begin
    # correctness of uncollect
    @test uncollect([1 2; 3 4], Symmetric(rand(2,2))) ≈ (data = [1 5; 0 4], uplo = nothing)
    @test uncollect(reshape(1:9,3,3), Symmetric(rand(3,3))) ≈ _uncollect(reshape(1:9,3,3), Symmetric(rand(3,3)))
    @test uncollect(reshape(1:9,3,3), Symmetric(rand(3,3), :L)) ≈ _uncollect(reshape(1:9,3,3), Symmetric(rand(3,3), :L))
end


