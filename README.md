# OddArrays.jl

This defines a few array types whose storage is quite different from their values:

```julia
julia> using OddArrays

julia> Rotation(pi/6) * Rotation(pi/6)  # stores one angle
2×2 Rotation{Float64}, with theta = 1.0471975511965976:
 0.5       -0.866025
 0.866025   0.5

julia> Vandermonde([0,2,10])  # coefficient vector, scale 1
3×3 Vandermonde{Int64, Vector{Int64}}:
 1   0    0
 1   2    4
 1  10  100

julia> Full(2pi, 1, 3)  # one number
1×3 Full{Float64, 2}:
 6.28319  6.28319  6.28319

julia> Range(1,2,3)  # start, stop, length::Int
3-element Range{Float64}, with step = 0.5:
 1.0
 1.5
 2.0

julia> Outer([5,7], [1,10,100])  # here two vectors, allows two matrices etc.
2×3 Outer{Int64, Vector{Int64}, Vector{Int64}}, storing 5 numbers:
 5  50  500
 7  70  700

julia> Mask([1 missing 3], [40,50,60]')  # two arrays
1×3 Mask{Int64, 2, Matrix{Union{Missing, Int64}}, Adjoint{Int64, Vector{Int64}}}:
 1  50  3

julia> PDiagMat([1,20,300])  # stores the inverse too
3×3 PDiagMat{Float64, Vector{Float64}}:
 1.0   0.0    0.0
 0.0  20.0    0.0
 0.0   0.0  300.0
```

They exist for the purpose checking that we know what we're doing with automatic differentiation. In particular, with reverse mode AD, there are issues of how to make sure the gradient stays inside the right subpace, and how best to represent it.

One problem we'd like to avoid is:

```julia
julia> using Zygote, ChainRulesCore

julia> gradient(_det, Vandermonde([3,4]))  # type's special definition, accesses fields
((coeff = [-1.0, 1.0], scale = 2.0),)

julia> gradient(prod, Vandermonde([3,4]))  # generic rule, makes a matrix
([12.0 4.0; 12.0 3.0],)

julia> gradient(x -> _det(x) / prod(x), Vandermonde([3,4]))  # without projection
ERROR: MethodError: no method matching +(::Matrix{Float64}, ::NamedTuple{(:coeff, :scale), ...})
```

There was a similar problem accumulating gradients for `Diagonal`:

```julia
julia> pullback(sum, Diagonal([3,-4]))[2](1.0)[1]
2×2 Fill{Float64}, with entries equal to 1.0

julia> pullback(x -> 5 * x.diag[1], Diagonal([3,-4]))[2](1)[1]
(diag = [5.0, 0.0],)
```

... which we can solve by standardising on the "natural" form, i.e., converting both contributions to `dx::Diagonal`:

```julia
julia> gradient(x -> sum(x) + 5 * x.diag[1], Diagonal([3,-4]))  # with Zygote#1104 + CRC#446 
2×2 Diagonal{Float64, Vector{Float64}}:
 6.0    ⋅ 
  ⋅    1.0
```

Perhaps these arrays which are nonlinear functions of their fields should instead standardise on the "structural" one?

## Structural gradients

The operation we want is the pullback of `collect`, which is called `uncollect` here. Some arrays have their own; the (slow but not wrong) fallback version uses Zygote's `pullback` and the array's own `getindex`.

```julia
julia> uncollect([0 1; 0 0], Vandermonde([3,4]))
[ Info: generic _uncollect
(coeff = [1.0, 0.0], scale = 3.0)
```

This is now called by `ProjectTo` for these arrays, which in turn is called by many generic rules, including the one for `prod`:

```julia
julia> gradient(x -> _det(x) / prod(x), Vandermonde([3,4]))
┌ Info: projecting to Tangent{Vandermonde}
└   typeof(dx) = Matrix{Float64}
[ Info: generic _uncollect
((coeff = [-0.1111111111111111, 0.0625], scale = -0.16666666666666666),)
```

Here `ProjectTo{OddArray}` saves the whole original array. Because in general the gradient subspace depends on the point.

While the possible perturbations of `θ` in `Rotation(θ)` are a 1-dimensional subspace of 2x2 matrices, the particular subspace depends on `θ`, etc. This is also why we cannot implement something like `+(::Matrix, ::Tangent{Rotation})` since, by that stage, the original `θ` has been lost.

```julia
julia> uncollect([0 1; 0 0], Rotation(pi/3))
(theta = -0.5000000000000001,)

julia> uncollect([0 1; 0 0], Rotation(pi/4))
(theta = -0.7071067811865476,)
```

Can this go wrong? We like "natural" `dx::Diagonal` so that it can flow backwards into generic rules. For this to matter, the original `x::Diagonal` must have been the output of a function which has a generic rule. Here, there are methods for multiplication of `r::Rotation`, which means one can be produced by `*` which has a generic rule. Which then fails, unless we opt out:

```julia
julia> gradient(x -> _getindex(x*x, 1,2), Rotation(pi/7))  # not good!
[ Info: *(::Rotation, ::Rotation)
ERROR: MethodError: no method matching *(::Tangent{Any, NamedTuple{(:theta,), Tuple{Float64}}}, ::Adjoint{Float64, Rotation{Float64}})

julia> gradient(x -> (_collect(Rotation(x))*_collect(Rotation(x)))[1,2], pi/7)  # desired result
(-1.2469796037174672,)

julia> ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::Rotation, ::Rotation)

julia> gradient(x -> _getindex(x*x, 1,2), Rotation(pi/7))
[ Info: *(::Rotation, ::Rotation)
((theta = -1.2469796037174672,),)
```

There is also a method `mul!(::Vector, ::Rotation, ::Vector)` which doesn't cause problems, since it never returns a `Rotation` matrix. The generic rule does make a full matrix before `uncollect` is called, and this can't be avoided by opting out:

```julia
julia> gradient(x -> (x * [1,0])[1], Rotation(pi/7))
[ Info: mul!(_, ::Rotation, _)
┌ Info: projecting to Tangent{Rotation}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
┌ Info: uncollect(_, ::Rotation)
└   theta = -0.4338837391175581
((theta = -0.4338837391175581,),)

julia> ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::Rotation, ::Vector)

julia> gradient(x -> (x * [1,0])[1], Rotation(pi/7))
[ Info: mul!(_, ::Rotation, _)
ERROR: Mutating arrays is not supported -- called setindex!(::Vector{Float64}, _...)
```

## Natural gradients

For some of these types, we can plausibly standardise on a "natural" gradient instead. Here we need other functions, mapping onto a different representation of the tangent space. 

The most trivial example is probably `Diagonal` (an honorary `OddArray`). The two new functions we need in general are, `restrict` & `naturalise`, are:

```julia
julia> restrict([1 2; 3 4], Diagonal)
2×2 Diagonal{Int64, Vector{Int64}}:
 1  ⋅
 ⋅  4

julia> uncollect([1 2; 3 4], Diagonal)
(diag = [1, 4],)

julia> naturalise(ans, Diagonal)
2×2 Diagonal{Int64, Vector{Int64}}:
 1  ⋅
 ⋅  4
```

These obey the following properties, all of them a bit trivial:

```julia
x = Diagonal(rand(3))
dx = rand(3,3); dx2 = randn(3,3)

@testset "simple naturalise checks for x::$(typeof(x).name.name)" begin
  # doesn't forget things which uncollect remembers:
  @test uncollect(naturalise(uncollect(dx, x), x), x) ≈ uncollect(dx, x)

  # linearity (using that of uncollect):
  @test naturalise(uncollect(33 * dx, x), x) ≈ 33 * naturalise(uncollect(dx, x), x)
  @test naturalise(uncollect(dx + dx2, x), x) ≈ naturalise(uncollect(dx, x), x) + naturalise(uncollect(dx2, x), x)

  # this defines restrict in terms of naturalise:
  @test restrict(dx, x) ≈ naturalise(uncollect(dx, x), x)
end
```

These are also satisfied by `x = Full(2,3,3)`, if the action of `restrict` & `naturalise` is this:

```julia
julia> uncollect([10 0 5], Full(pi,1,3))
(value = 15, size = nothing)  # sum

julia> restrict([10 0 5], Full)  # doesn't depend on the point, just type
1×3 Full{Float64, 2}:
 5.0  5.0  5.0                # mean

julia> naturalise((value = 15, size = nothing), Full(pi,1,3))  # needs the size
1×3 Full{Float64, 2}:
 5.0  5.0  5.0                # value / length
```

What those tests don't check is that `naturalise` maps into the cotangent subspace corresponding to the submanifold defined by the type.
This is a bit more involved to check, but is the argument against this returning say `[15 0 0]`, which is in the same equivalence class as `[10 0 5]` and `[5 5 5]` according to `uncollect`.
You can try the above tests with `x = Full(2.0, 3, 3; one=true)`, but the new check is this:

```julia
julia> restrict([10 0 5], Full(pi, 1, 3; one=true))
1×3 OneElement(::Int64):
 15  0  0

julia> subspacetest(Full, 2.0, 3, 3; one=true);
┌ Warning: naturalise(dw, Full) has components in 8 directions outside T*S
└ @ OddArrays ~/.julia/dev/OddArrays/src/OddArrays.jl:958

julia> subspacetest(Full, 2.0, 3, 3);
[ Info: naturalise(dw, Full) seems to live in T*S, as it should
```

Less obviously, there is a correct projection for `Range` objects:

```julia
julia> uncollect([1,1,13], Range(1,2,3))
(start = 1.5, stop = 13.5, len = nothing)

julia> naturalise(ans, Range(1,2,3))
3-element Range{Float64}, with step = 6.0:
 -1.0
  5.0
 11.0

julia> ans ≈ restrict([1,1,13], Range)
true

julia> x = Range(0,ℯ,5); dx = rand(5); dx2 = rand(5);  # for above @testset

julia> subspacetest(Range, 1.2, 3.4, 5);
[ Info: naturalise(dw, Range) seems to live in T*S, as it should
```

For rotation matrices, an example of `restrict` which passes the above tests but fails `subspacetest` is this:

```julia
julia> restrict([1 2; 3 4], Rotation(pi/3))
2×2 AntiSymOne{Float64}:
  0.0      3.83013
 -3.83013  0.0

julia> x = Rotation(randn()); dx = randn(2,2); dx2 = randn(2,2);  # for above @testset

julia> subspacetest(Rotation, pi/3);
┌ Warning: naturalise(dw, Rotation) has components in 3 directions outside T*S
└ @ OddArrays ~/.julia/dev/OddArrays/src/OddArrays.jl:958

```

It's pretty that the cotangent lives in the Lie algebra, but in fact irrelevant.
The way to stay inside the submanifold is to use the dual part of this, which you could represent as a scaled rotation matrix:

```julia
julia> using ForwardDiff: Dual

julia> Rotation(Dual(pi/3, 1000))
2×2 Rotation{Dual{Nothing, Float64, 1}}, with theta = Dual{Nothing}(1.0471975511965976,1000.0):
 Dual{Nothing}(0.5,-866.025)    Dual{Nothing}(-0.866025,-500.0)
 Dual{Nothing}(0.866025,500.0)   Dual{Nothing}(0.5,-866.025)
```

This can't be a `Rotation` struct, in fact that's obvious from the start as the cotangent representation has to be a vector space, but the sum of two rotation matrices is outside the set.


## Over-parameterised types

These store more numbers than there are dimensions in the matrix subspace. They have unambiguous "structural" gradients:

```julia
julia> gradient(x -> x[1], UnitVector([3,0,4]))
┌ Info: projecting to Tangent{UnitVector}
└   typeof(dx) = OneElement{Float64, 1, Tuple{Int64}, Tuple{Base.OneTo{Int64}}}
[ Info: generic _uncollect
((raw = [0.128, 0.0, -0.096],),)

julia> gradient(x -> x[1], Outer(3, [4 5; 6 7]))
┌ Info: projecting to Tangent{Outer}
└   typeof(dx) = Matrix{Int64} (alias for Array{Int64, 2})
((x = 4, y = [3 0; 0 0], size = nothing),)

julia> gradient(x -> sum(abs2, x), Mask([1,NaN,3], [40,50,60]))  # with default OddArray projection
┌ Info: projecting to Tangent{Mask}
└   typeof(dx) = Vector{Float64} (alias for Array{Float64, 1})
((alpha = [2.0, 0.0, 6.0], beta = [0.0, 100.0, 0.0]),)
```

Do they have "natural" ones? For `Mask` can you just add `alpha + beta`:

```julia
julia> restrict([1 2 3], Mask)
1×3 Matrix{Int64}:
 1  2  3

julia> naturalise((alpha = [2 0 6], beta = [0 100 0]), Mask)
1×3 Matrix{Int64}:
 2  100  6

julia> x = Mask([1,NaN,3], [40,50,60]); dx = randn(3); dx2 = randn(3);  # for above @testset

julia> subspacetest(Mask, [1,NaN,3], [40,50,60]);
[ Info: naturalise(dw, Mask) seems to live in T*S, as it should

julia> gradient(x -> x.beta[1]^2, x)  # reads a field which doesn't contribute... garbage primal?
([80.0, 0.0, 0.0],)
```

For `Outer`, there is more serious redundancy, `Outer([4], [9,9]) == Outer([6], [6,6]) == Outer([9], [4,4])` describe the matrix `x`. And the constructor is nonlinear.
You can still make a valid `naturalise`, I think, but it's not trivial and it cannot in general re-use the struct:

```julia
julia> uncollect([3 0 0; 0 0 0], Outer([5,5], [7,7,7]))  # S is 4 dimensional here
(x = [21, 0], y = [15, 0, 0], size = nothing)

julia> naturalise(ans,  Outer([5,5], [7,7,7]))  # this cannot be written as Outer
2×3 Matrix{Float64}:
 2.0   0.5   0.5
 1.0  -0.5  -0.5

julia> uncollect(ans, Outer([5,5], [7,7,7]))
(x = [21.0, -8.881784197001252e-16], y = [15.0, -8.881784197001252e-16, 0.0], size = nothing)

julia> subspacetest(Outer, [5,6], [7,8,9]);
[ Info: naturalise(dw, Outer) seems to live in T*S, as it should
```

The case `Outer(::Matrix, ::Number)` is simpler:

```julia
julia> uncollect([1 10 100], Outer([4 5 6], 7))
(x = [7 70 700], y = 654, size = nothing)

julia> naturalise(ans, Outer([4 5 6], 7))
1×3 Matrix{Float64}:
 1.0  10.0  100.0

julia> ans == Outer([1 10 100], 1) == Outer([2 20 200], 1/2)  # but no advantage
true
```

Next, `PDiagMat` stores both the diagonal and its inverse. It specialises `*` of two such to produce a third, and opts out of the generic rule:

```julia
julia> gradient(x -> (x * x)[5], PDiagMat([1,2,3]))
[ Info: *(::PDiagMat, ::PDiagMat)
┌ Info: projecting to Tangent{PDiagMat}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
((dim = nothing, diag = [0.0, 4.0, 0.0], inv_diag = nothing),)

julia> gradient(x -> (x * _inv(x))[5], PDiagMat([1,2,3]))  # weird, uncollect could never make this
[ Info: _inv(::PDiagMat)
[ Info: *(::PDiagMat, ::PDiagMat)
┌ Info: projecting to Tangent{PDiagMat}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
((dim = nothing, diag = [0.0, 0.5, 0.0], inv_diag = [0.0, 2.0, 0.0]),)

julia> gradient(x -> (PDiagMat(x) * _inv(PDiagMat(x)))[5], [1,2,3])
[ Info: _inv(::PDiagMat)
[ Info: *(::PDiagMat, ::PDiagMat)
┌ Info: projecting to Tangent{PDiagMat}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
([0.0, 0.0, 0.0],)
```

Haven't sorted these ones out.

## Discussed elsewhere

This PR https://github.com/JuliaDiff/ChainRulesCore.jl/pull/449 contains some comparable maps. (Formatted [notes.md](https://github.com/JuliaDiff/ChainRulesCore.jl/blob/wct/writing-generic-rrules/notes.md) and [examples](https://github.com/JuliaDiff/ChainRulesCore.jl/blob/wct/writing-generic-rrules/examples.jl).)

* Since `destructure == collect`, the useful map from Matrix to Tangent is called `destructure_pullback` or else `pullback_of_destructure(x)(dx)` for `uncollect(dx, x)` here.

* There is also a "Restructure", and I think `pullback_of_restructure` is playing a role like `naturalise` here. But I am not very sure.

* The `ScaledVector` example there is much like `Outer(pi, [0 1 2])` here, but `Outer` allows other things.

