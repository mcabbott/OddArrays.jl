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

The operation we want is the pullback of `collect`, which is called `uncollect` here. Some arrays have their own; the (slow but not wrong) fallback version uses Zygote's `pullback` and the array's own `getinidex`.

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

# same subspace as uncollect sees:
@test uncollect(naturalise(uncollect(dx, x), x), x) ≈ uncollect(dx, x)

# linearity (using that of uncollect):
@test naturalise(uncollect(33 * dx, x), x) ≈ 33 * naturalise(uncollect(dx, x), x)
@test naturalise(uncollect(dx + dx2, x), x) ≈ naturalise(uncollect(dx, x), x) + naturalise(uncollect(dx2, x), x)

# this defines restrict:
@test restrict(dx, x) ≈ naturalise(uncollect(dx, x), x)
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

Is there any freedom here? `naturalise` has access to only one number. If it makes a `Full`, the scale is fixed. But why should it not produce say `[15 0 0]`? This is in the same equivalence class as `[10 0 5]` and `[5 5 5]` according to `uncollect`.

You can try this out as `Full(2,3,3; one=true)`. Is this just unaesthetic or will it ever give wrong answers? Tests pass. And what exactly is the mathematical condition which would forbid this, if we wanted to --- that we stay somehow within the natural embedding of the subspace?


For some of the types above, the constructor (from fields to an array) is not linear, so cannot be part of `restrict` or `naturalise`. It is nevertheless possible to derive a natural representation, at least sometimes. For example:

```julia
julia> restrict([1 2; 3 4], Rotation(pi/3))
2×2 AntiSymOne{Float64}:
  0.0      3.83013
 -3.83013  0.0

julia> x = Rotation(randn()); dx = randn(2,2); dx2 = randn(2,2);  # for above tests
```

It's pretty that the cotangent lives in the Lie algebra, although really we aren't much involved in matrix multiplication here. It's not clear this is a particularly natural choice --- we could also pick a scaled rotation matrix, like the dual part of this:

```julia
julia> using ForwardDiff: Dual

julia> Rotation(Dual(pi/3, 1000))
2×2 Rotation{Dual{Nothing, Float64, 1}}, with theta = Dual{Nothing}(1.0471975511965976,1000.0):
 Dual{Nothing}(0.5,-866.025)    Dual{Nothing}(-0.866025,-500.0)
 Dual{Nothing}(0.866025,500.0)   Dual{Nothing}(0.5,-866.025)
```

Anyway, we could use `ProjectTo` to standardise on one of these instead of the `Tangent`. But it doesn't seem very useful to do so. While the forward pass of `*(::Rotation, ::Rotation)` is just addition, the gradient will tend to involve `*(::AntiSymOne, ::Rotation)` for which there are no pre-existing optimisations. It'll work, but via the most generic methods.

Maybe that's the rule of thumb for all types where the embedding map is nonlinear in the fields?


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

Do they have "natural" ones? For `Mask` can you just add `alpha + beta`... can this go wrong?

```julia
julia> restrict([1 2 3], Mask)
1×3 Matrix{Int64}:
 1  2  3

julia> naturalise((alpha = [2 0 6], beta = [0 100 0]), Mask)
1×3 Matrix{Int64}:
 2  100  6

julia> x = Mask([1,NaN,3], [40,50,60]); dx = randn(3); dx2 = randn(3);  # for tests above

julia> gradient(x -> x.beta[1]^2, x)  # reads a field which doesn't contribute... garbage primal?
([80.0, 0.0, 0.0],)
```

For `Outer`, here are two representations of the same point, i.e. different fields describing the same matrix `x`. Can we prove this has no legal `naturalise` at all? 

```julia
julia> Outer([5], [7]) == Outer([7], [5])
true

julia> uncollect([3;;], Outer([5], [7]))
(x = [21], y = [15], size = nothing)

julia> uncollect([3;;], Outer([7], [5]))
(x = [15], y = [21], size = nothing)

julia> uncollect([3 0 0; 0 0 0], Outer([5,5], [7,7,7]))
(x = [21, 0], y = [15, 0, 0], size = nothing)

julia> sqrt.([21, 0] .* [15, 0, 0]' ./ [5,5] ./ [7,7,7]')  # nonlinear
2×3 Matrix{Float64}:
 3.0  0.0  0.0
 0.0  0.0  0.0
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

This PR https://github.com/JuliaDiff/ChainRulesCore.jl/pull/449 contains some comparable maps, with more confusing names. (Formatted [notes.md](https://github.com/JuliaDiff/ChainRulesCore.jl/blob/wct/writing-generic-rrules/notes.md) and [examples](https://github.com/JuliaDiff/ChainRulesCore.jl/blob/wct/writing-generic-rrules/examples.jl).) I'm ignoring anything to do with forward mode entirely, "tangent" not "cotangent" there.

* Since `destructure == collect`, the useful map from Matrix to Tangent is called `destructure_pullback` or else `pullback_of_destructure(x)(dx)` for `uncollect(dx, x)` here.

* There is also a "Restructure", and I think `pullback_of_restructure` is playing a role like `naturalise` here. But I am not very sure. Is the claim for `Fill` that its scale is arbitrary?

* The `ScaledVector` example there is much like `Outer(pi, [0 1 2])` here, but `Outer` allows other things.

In addition to this story about subspaces and representations, there's a specific proposal to replace the fairly permissive behaviour of ChainRules (natural or structural both allowed by default, generic rules assume the former) with one which is all-Tangent all the time. I called that "the nuclear option" in https://github.com/JuliaDiff/ChainRulesCore.jl/issues/441 .

