# OddArrays.jl

This defines a few array types whose storage is quite different from their values:

```julia
julia> using OddArrays

julia> Rotation(pi/3) * Rotation(pi/3)  # one angle
2×2 Rotation{Float64}:
 0.5       -0.866025
 0.866025   0.5

julia> LogRange(1,2,3)  # start, stop, length
3-element LogRange{Float64}:
 1.0
 1.4142135623730951
 2.0

julia> Vandermonde([0,2,10])  # coefficient vector, scale 1
3×3 Vandermonde{Int64, Vector{Int64}}:
 1   0    0
 1   2    4
 1  10  100

julia> Outer([5,7], [1,10,100])  # here two vectors, allows two matrices etc.
2×3 Outer{Int64, Vector{Int64}, Vector{Int64}}:
 5  50  500
 7  70  700

julia> Full(2pi, 1, 3)  # one number
1×3 Full{Float64, 2}:
 6.28319  6.28319  6.28319

julia> Mask([1 missing 3], [40,50,60]')  # two arrays
1×3 Mask{Int64, 2, Matrix{Union{Missing, Int64}}, Adjoint{Int64, Vector{Int64}}}:
 1  50  3
```

They exist for the purpose checking that we know what we're doing with automatic differentiation.

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

Without projection, there was a similar problem accumulating gradients for `Diagonal`, one which actually showed up in the wild:

```julia
julia> pullback(sum, Diagonal([3,-4]))[2](1.0)[1]
2×2 Fill{Float64}, with entries equal to 1.0

julia> pullback(x -> 5 * x.diag[1], Diagonal([3,-4]))[2](1)[1]
(diag = [5.0, 0.0],)
```

... which we can solve by standardising on the "natural" form. Perhaps these arrays should standardise on the "structural" one?

The operation to do so is the pullback of `collect`, which is called `decollect` here. Some arrays have their own; the (slow but not wrong) fallback version uses Zygote's `pullback` and the array's own `getinidex`. This is now called by `ProjectTo` for these arrays, which in turn is called by many generic rules:

```julia
julia> gradient(x -> sum(x) + 5 * x.diag[1], Diagonal([3,-4]))  # with Zygote#1104 + CRC#446 
2×2 Diagonal{Float64, Vector{Float64}}:
 6.0    ⋅ 
  ⋅    1.0

julia> gradient(x -> _det(x) / prod(x), Vandermonde([3,4]))
┌ Info: projecting to Tangent{Vandermonde}
└   typeof(dx) = Matrix{Float64}
[ Info: generic _decollect
((coeff = [-0.1111111111111111, 0.0625], scale = -0.16666666666666666),)
```

Here `ProjectTo{OddArray}` saves the whole original array. Because, while the possible perturbations of `θ` in `Rotation(θ)` are a 1-dimensional subspace of 2x2 matrices, the particular subspace depends on `θ`, etc. This is also why we cannot implement something like `+(::Matrix, ::Tangent{Rotation})` since, by that stage, the original `θ` has been lost.
```julia
julia> decollect([0 1; 0 0], Rotation(pi/3))
(theta = -0.5000000000000001,)

julia> decollect([0 1; 0 0], Rotation(pi/4))
(theta = -0.7071067811865476,)
```

Can this go wrong? We like "natural" `Diagonal` so that it can flow backwards into generic rules. There are no special rules for `r::Rotation`, but there are methods for multiplication. Which means that `r` can be produced by an operation which has a generic rule. Which fails unless we opt out of the generic rule:

```julia
julia> gradient(x -> _getindex(x*x, 1,2), Rotation(pi/7))  # not good! but why no error?
[ Info: *(::Rotation, ::Rotation)
((theta = [-1.1234898018587336 -0.5410441730642656; 0.5410441730642656 -1.1234898018587336],),)

julia> gradient(x -> (x*x)[1,2], Rotation(pi/7))
[ Info: *(::Rotation, ::Rotation)
┌ Info: projecting to Tangent{Rotation}
└   typeof(dx) = Zygote.OneElement{Float64, 2, Tuple{Int64, Int64}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
┌ Info: decollect(_, ::Rotation)
└   theta = -0.6234898018587336
((theta = [-1.1234898018587336 -0.5410441730642656; 0.5410441730642656 -1.1234898018587336],),)

julia> gradient(x -> (_collect(Rotation(x))*_collect(Rotation(x)))[1,2], pi/7)  # desired result
(-1.2469796037174672,)

julia> ChainRulesCore.@opt_out rrule(::typeof(*), ::Rotation, ::Rotation)
rrule (generic function with 1 method)  # umm

julia> ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::Rotation, ::Rotation)

julia> gradient(x -> _getindex(x*x, 1,2), Rotation(pi/7))
[ Info: *(::Rotation, ::Rotation)
((theta = -1.2469796037174672,),)
```

There is also a method `mul!(::Vector, ::Rotation, ::Vector)` which doesn't cause problems, since it never returns a `Rotation` matrix. The generic rule does make a full matrix before `decollect` is called, and this can't be avoided by opting out:

```julia
julia> gradient(x -> (x * [1,0])[1], Rotation(pi/7))
[ Info: mul!(_, ::Rotation, _)
┌ Info: projecting to Tangent{Rotation}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
┌ Info: decollect(_, ::Rotation)
└   theta = -0.4338837391175581
((theta = -0.4338837391175581,),)

julia> ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::Rotation, ::Vector)

julia> gradient(x -> (x * [1,0])[1], Rotation(pi/7))
[ Info: mul!(_, ::Rotation, _)
ERROR: Mutating arrays is not supported -- called setindex!(::Vector{Float64}, _...)
```

Other examples which work (without needing projection):

```julia
julia> gradient(x -> x[1,1], Outer(3, [4 5; 6 7]))  # over-parameterised
┌ Info: projecting to Tangent{Outer}
└   typeof(dx) = Zygote.OneElement{Int64, 2, Tuple{Int64, Int64}, Tuple{Base.OneTo{Int64}, Base.OneTo{Int64}}}
((x = 4, y = [3 0; 0 0], size = nothing),)

julia> gradient(x -> sum(abs2, x), Mask([1,NaN,3], [40,50,60]))
┌ Info: projecting to Tangent{Mask}, II
└   typeof(dx) = Vector{Float64} (alias for Array{Float64, 1})
((alpha = [2.0, 0.0, 6.0], beta = [0.0, 100.0, 0.0]),)

julia> gradient(x -> _getindex(x,2), LogRange(1,2,3))  # should projection kill `len`?
((start = 0.7071067811865476, stop = 0.3535533905932738, len = -0.2450645358671368),)
```
