"""
    module OddArrays

This defines some array types, with more complicated relation
between storage and elements than the ones in LinearAlgebra.
Incomplete list:

* `Rotation`
* `Vandermonde`
* `Outer`
* `Mask`
* `Full`
* `Range`

Call `_getindex`, `_collect` or `_getfield` on any of them
to bypass the Base functions, and hence the generic gradient definitions.

Some define their own `_sum`, `_prod`, `_det`, `_mul`, which likewise have no
gradient definitions.

Many things are done inefficiently, and some operations print `@info` to
explain what they are doing.
"""
module OddArrays
export OddArray, UnitVector, Rotation, AntiSymOne, LogRange, Vandermonde, Outer, Mask, Full, Alphabet, Range, PDiagMat,
    ONEGRAD, TangentOrNT, uncollect, _uncollect, naturalise, restrict,
    _getindex, _collect, _getfield, _mul, _det, _prod, _sum, _inv

using Statistics, LinearAlgebra

using ChainRulesCore, Zygote
using Zygote: OneElement
export ProjectTo, Tangent, NoTangent, ZeroTangent, rrule,  # CRC
    Zygote, gradient, pullback, OneElement,  # Zygote
    Diagonal, UpperTriangular, mean

abstract type OddArray{T,N} <: AbstractArray{T,N} end

function Base.NamedTuple(x::OddArray)
    F = fieldnames(typeof(x))
    NamedTuple{F}(map(f -> getfield(x, f), F))
end

#==== gradient stuff =====#

_collect(x::AbstractArray) = reshape([_getindex(x, I.I...) for I in CartesianIndices(size(x))], size(x))
Base.collect(x::OddArray) = _collect(x)  # not sure I need this
ChainRulesCore.@non_differentiable CartesianIndices(::Tuple)  # latest CR has this too

_getfield(x, s::Symbol) = getfield(x, s)
# This differs in that it has a gradient without projection,
# needed for testing collect with types where projection would otherwise
# turn a Tangent back to an array:
function ChainRulesCore.rrule(::typeof(_getfield), x::T, s::Symbol) where {T}
    function un_getfield(dy)
        nots = NamedTuple{fieldnames(T)}(map(Returns(NoTangent()), fieldnames(T)))
        (NoTangent(), Tangent{T}(; nots..., s => dy), NoTangent())
    end
    getfield(x, s), un_getfield
end

const TangentOrNT = Union{Tangent, NamedTuple}

# In easy cases, these functions only need the type:
naturalise(dx::TangentOrNT, x::OddArray) = naturalise(dx, typeof(x))
restrict(dx::AbstractArray, x::OddArray) = size(dx) == size(x) ? restrict(dx, typeof(x)) : throw("dimension mismatch!")

"""
    uncollect(dx::Array, x::OddArray) -> NamedTuple
    uncollect(dx::Array, ::Type{OddArray}) -> NamedTuple

This turns a natural into a structural representation.
Called by `ProjectTo{OddArray}(::Array)` by default.

Fallback implementation is `_uncollect`, which calls `pullback(_collect, x)`
using Zygote, and discards its forward pass. And `_collect` calls `_getindex` 
which ideally calls `_getfield`, bypassing Base's functions which have
generic rules.

The fallback one has to know `x`. For some array types,
the hand-written version needs only `typeof(x)`.
"""
uncollect(dx::AbstractArray, x::AbstractArray) = _uncollect(dx, x)
function _uncollect(dx::AbstractArray, x::AbstractArray)
    _, back = Zygote.pullback(_collect, x)
    @info "generic _uncollect"
    back(dx)[1]
end

ChainRulesCore.ProjectTo(x::OddArray) = ProjectTo{OddArray}(; x = x)
function (p::ProjectTo{OddArray})(dx::AbstractArray)
    @info "projecting to Tangent{$(typeof(p.x).name.name)}" typeof(dx)
    Tangent{typeof(p.x)}(; uncollect(dx, p.x)...)
end

(p::ProjectTo{OddArray})(dx::Tangent) = dx # Zygote makes Tangent{Any}, unfortunately

#####
##### Array types!
#####

"""
    Rotation(θ)

This is a 2x2 rotation matrix, parameterised by the angle.
Has an optimised method for multiplication with a vector, via `mul!`,
multiplication of two rotations, `*` via `_mul`, 
and `inv`erse via `_inv`.
"""
struct Rotation{T} <: OddArray{T,2}
    theta::T
    Rotation(theta::T) where {T<:Number} = new{float(T)}(float(theta))
end

Base.size(r::Rotation) = (2,2)

Base.getindex(r::Rotation, i::Int, j::Int) = _getindex(r, i, j)
function _getindex(r::Rotation, i, j)
    checkbounds(r, i, j)
    theta = _getfield(r, :theta)
    if i==j
        return cos(theta)
    else
        s = sin(theta)
        return i>j ? s : -s
    end
end

Base.showarg(io::IO, r::Rotation, top=true) = print(io, "Rotation{", eltype(r), "}", top ? ", with theta = $(r.theta)" : "")

#==== optimisations of generic functions =====#

Base.:*(r::Rotation, s::Rotation) = _mul(r, s)
function _mul(r::Rotation, s::Rotation)
    _info("_mul(::Rotation, ::Rotation)")
    Rotation(r.theta + s.theta)
end

Base.inv(r::Rotation) = _inv(r)
function _inv(r::Rotation)
    _info("_inv(::Rotation)")
    Rotation(-r.theta)
end

_info(s::String) = @info s
Zygote.@nograd _info

function LinearAlgebra.mul!(y::AbstractVector, r::Rotation, x::AbstractVector)
    _info("mul!(_, ::Rotation, ::Vector)")
    s, c = sincos(r.theta)
    x1, x2 = x
    y[1] = c * x1 - s * x2
    y[2] = s * x1 + c * x2
    y
end

#==== gradient stuff =====#

function uncollect(dx::AbstractMatrix, r::Rotation)
    s, c = sincos(r.theta)
    theta = - s * dx[1,1] - c * dx[1,2] + c * dx[2,1] - s * dx[2,2]
    @info "uncollect(_, ::Rotation)" theta
    (; theta)
end

function restrict(dx::AbstractMatrix, r::Rotation)
    s, c = sincos(r.theta)
    theta = - s * dx[1,1] - c * dx[1,2] + c * dx[2,1] - s * dx[2,2]  # same as uncollect
    AntiSymOne(theta / (2c))
end

function naturalise(dx::TangentOrNT, r::Rotation)
    AntiSymOne(dx.theta / (2 * cos(r.theta)))
end

"""
    AntiSymOne(φ)

This is a 2x2 antisymmetric matrix.
`exp` maps this to a `Rotation` matrix.
"""
struct AntiSymOne{T} <: OddArray{T,2}
    phi::T
    AntiSymOne(phi::T) where {T<:Number} = new{T}(phi)
end
Base.size(s::AntiSymOne) = (2,2)
Base.getindex(s::AntiSymOne, i::Int, j::Int) = _getindex(s, i, j)
_getindex(s::AntiSymOne, i::Int, j::Int) = i==j ? zero(eltype(s)) : i>j ? s.phi : -s.phi

Base.exp(s::AntiSymOne) = Rotation(s.phi)

Base.:+(s::AntiSymOne, s2::AntiSymOne) = AntiSymOne(s.phi + s2.phi)
Base.:*(s::AntiSymOne, λ::Number) = AntiSymOne(s.phi * λ)
Base.:*(λ::Number, s::AntiSymOne) = AntiSymOne(λ * s.phi)

Base.:*(s::AntiSymOne, t::AntiSymOne) = _mul(s, t)
function _mul(s::AntiSymOne, t::AntiSymOne)
    _info("_mul(::AntiSymOne, ::AntiSymOne)")
    val = - s.phi * t.phi
    Diagonal([val, val])
end

"""
    UnitVector(v)

Parameterises a vector `u` of length `n` with `norm(u) == 1` by another vector of length `n`.
Lives in a subspace of dimension `n-1`.
```
julia> u = UnitVector(0:3:3)
2-element UnitVector{Float64, StepRange{Int64, Int64}}, with raw norm = 3.0:
 0.0
 1.0

julia> gradient(norm, u)  # opts out of generic
(nothing,)

julia> gradient(sum, u)
┌ Info: projecting to Tangent{UnitVector}
└   typeof(dx) = FillArrays.Fill{Float64, 1, Tuple{Base.OneTo{Int64}}}
[ Info: generic _uncollect
((raw = [0.3333333333333333, 0.0],),)
```
"""
struct UnitVector{T,S} <: OddArray{T,1}
    raw::S
    UnitVector(vec::S) where {S<:AbstractVector{T}} where {T<:Number} = new{float(T),S}(vec)
end
Base.parent(u::UnitVector) = _getfield(u, :raw)

Base.size(u::UnitVector) = size(parent(u))

Base.getindex(u::UnitVector, i::Int) = _getindex(u, i)
_getindex(u::UnitVector, i) = parent(u)[i] / norm(parent(u))

Base.showarg(io::IO, u::UnitVector, top=true) = print(io, typeof(u), top ? ", with raw norm = $(norm(parent(u)))" : "")

#==== optimisations of generic functions =====#

LinearAlgebra.norm2(u::UnitVector) = one(eltype(u))

Base.sum(u::UnitVector) = _sum(u)
_sum(u::UnitVector) = sum(parent(u)) / norm(parent(u))

#==== gradient stuff =====#

ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(norm), ::UnitVector)
ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(norm), ::UnitVector, p::Real)
ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(LinearAlgebra.norm2), ::UnitVector)

"""
    OneElement(val, ind::Tuple, axes::Tuple)
    OneElement(array)

Borrowed from Zygote, but with a freindlier constructor added.
Honourary OddArray, i.e. defines `_getindex` etc.
```
julia> OneElement([0 -33 1e-16 eps()])  # keeps largest abs
1×4 OneElement(::Float64):
 0.0  -33.0  0.0  0.0

julia> uncollect([1 10 100 1000], ans)
[ Info: generic _uncollect
(val = 10, ind = nothing, axes = nothing)
```
"""
function Zygote.OneElement(x::AbstractArray)
    _, i = findmax(abs, x)
    OneElement(x[i], Tuple(i), axes(x))
end

_getindex(o::OneElement{T,N}, ind::Vararg{Int,N}) where {T,N} = ind==o.ind ? _getfield(o,:val) : zero(T)

Base.showarg(io::IO, o::OneElement, top=true) = print(io, "OneElement(::", eltype(o), ")")


"""
    LogRange(α, ω, ℓ=5)

Like `range(α, ω, length=ℓ)` except geometrically not linearly spaced,
i.e. constant ratio not difference.
Has an optimised method for `prod`, exported as `_prod`.
```
julia> LogRange(1,2,3)  # start, stop, length::Int
3-element LogRange{Float64}:
 1.0
 1.4142135623730951
 2.0
```
"""
struct LogRange{T} <: OddArray{T,1}
    start::T
    stop::T
    len::Int
    LogRange(start::T1, stop::T2, len::Integer) where {T1<:Number, T2<:Number} = new{float(promote_type(T1,T2))}(start, stop, len)
end
LogRange(start::Number, stop::Number; length::Integer=5) = LogRange(start, stop, length)

Base.size(l::LogRange) = (l.len,)

Base.getindex(l::LogRange, i::Int) = _getindex(l, i)
_getindex(l::LogRange, i) = l.start * (l.stop/l.start)^((i-1)/(l.len-1))

Base.showarg(io::IO, l::LogRange, top=true) = print(io, "LogRange{", eltype(l), "}", top ? ", with ratio = $((l.stop/l.start)^(1/(l.len-1)))" : "")

#==== optimisations of generic functions =====#

Base.prod(l::LogRange) = _prod(l)
_prod(l::LogRange) = (l.start * l.stop)^(l.len/2)

"""
    Vandermonde(x, s=1)

Defines an NxN matrix given a vector of N coefficients,
whose values are `hcat((x .^ i for i in 0:N-1)...) .* s`
with overall scale `s`.

Has an optimised method for `det`, exported as `_det`.
```
julia> Vandermonde([0,2,3,10], 1.001f0)  # 5 parameters for 16 entries
4×4 Vandermonde{Float64, Vector{Int64}}:
 1.001   0.0      0.0       0.0
 1.001   2.002    4.004     8.008
 1.001   3.003    9.009    27.027
 1.001  10.01   100.1    1001.0

julia> gradient(_det, ans)
((coeff = [-3148.5628285471344, -2108.412608402099, 4016.0240160039975, 1240.9514209452354], scale = 13480.360333439994),)
```
"""
struct Vandermonde{T,V} <: OddArray{T,2}
    coeff::V
    scale::T
    function Vandermonde(coeff::V, scale::S=true) where {V<:AbstractVector, S<:Number}
        T = promote_type(eltype(V), S)
        new{T,V}(coeff, scale)
    end
end

Base.size(m::Vandermonde) = (length(m.coeff), length(m.coeff))

Base.getindex(m::Vandermonde, i::Int, j::Int) = _getindex(m, i, j)
function _getindex(m::Vandermonde, i, j)
    checkbounds(m, i, j)
    m.coeff[i] ^ (j-1) * m.scale
end

#==== optimisations of generic functions =====#

LinearAlgebra.det(m::Vandermonde) = _det(m)
# _det(m::Vandermonde) = prod(float(m.coeff[j] - m.coeff[i]) for j in 1:length(m.coeff) for i in 1:j-1) * m.scale ^ length(m)
function _det(m::Vandermonde)
    out = float(m.scale) ^ length(m.coeff)
    for j in 1:length(m.coeff)
        for i in 1:j-1
            out *= (m.coeff[j] - m.coeff[i])
        end
    end
    out
end

"""
    Outer(x,y) ≈ x * y'

Lazy outer product matrix.

If `x, y` are vectors then this has 2N parameters for N^2 entries,
but describes a subspace of dimension 2N-1.

With two matrices, or a matrix and a scalar, this can have more than N^2 parameters.
```
julia> Outer([0 1 2], pi)
1×3 Outer{Float64, Matrix{Int64}, Irrational{:π}}, storing 4 numbers:
 0.0  3.14159  6.28319

julia> Outer([10,20], [3,4,5])
2×3 Outer{Int64, Vector{Int64}, Vector{Int64}}, storing 5 numbers:
 30  40   50
 60  80  100

julia> ans == Outer([1,2], [30,40,50])
true
```
"""
struct Outer{T,X,Y} <: OddArray{T,2}
    x::X
    y::Y
    size::Tuple{Int,Int}
    function Outer(x::X, y::Y) where {X,Y}
        z = x * y'
        z isa AbstractMatrix || throw("expected that x * y' be a matrix")
        new{eltype(z),X,Y}(x, y, size(z))
    end
end

Base.size(o::Outer) = o.size

Base.getindex(o::Outer, i::Int, j::Int) = _getindex(o, i, j)
_getindex(o::Outer, i, j) = (o.x * o.y')[i, j]
_getindex(o::Outer{<:Any,<:AbstractVector,<:AbstractVector}, i, j) = o.x[i] * o.y[j]'
_getindex(o::Outer{<:Any,<:AbstractMatrix,<:Number}, i, j) = o.x[i,j] * o.y'
_getindex(o::Outer{<:Any,<:Number,<:AbstractMatrix}, i, j) = o.x * o.y[j,i]'
_getindex(o::Outer{<:Any,<:AbstractMatrix,<:AbstractMatrix}, i, j) = sum(o.x[i,k] * o.y[j,k]' for k in axes(o.x,2))

Base.showarg(io::IO, o::Outer, top=true) = print(io, typeof(o), top ? ", storing $(length(o.x) + length(o.y)) numbers" : "")

#==== gradient stuff =====#

uncollect(dz::AbstractMatrix, o::Outer{<:Any,<:AbstractVector,<:AbstractVector}) =
    (; x = dz * o.y, y = dz' * o.x, size=nothing)
uncollect(dz::AbstractMatrix, o::Outer{<:Any,<:AbstractMatrix,<:Number}) =
    (; x=dz .* o.y, y=dot(dz, o.x), size=nothing)  # wrong for complex numbers?
uncollect(dz::AbstractMatrix, o::Outer{<:Any,<:Number,<:AbstractMatrix}) =
    (; x=dot(dz, o.y'), y=dz' .* o.x, size=nothing)  # wrong for complex numbers?

"""
    Mask(α, β)

Where `α` is finite, and not `missing` or `nothing`, its values are used.
Otherwise, the values of `β` at the same index.
"""
struct Mask{T,N,X,Y} <: OddArray{T,N}
    alpha::X
    beta::Y
    function Mask(alpha::X, beta::Y) where {X<:AbstractArray, Y<:AbstractArray}
        T = mask_promote(eltype(X), eltype(Y))
        size(alpha) == size(beta) || throw("sizes must agree")
        new{T,ndims(alpha),X,Y}(alpha, beta)
    end
end
mask_promote(T, S) = promote_type(mask_promote(T), mask_promote(S))
mask_promote(::Type{Union{Missing, T}}) where {T} = T
mask_promote(::Type{Union{Nothing, T}}) where {T} = T
mask_promote(T) = T

Base.size(m::Mask) = size(_getfield(m, :alpha))

Base.getindex(m::Mask, ijk::Int...) = _getindex(m, ijk...)
function _getindex(m::Mask{T}, ijk...) where {T}
    a = _getfield(m, :alpha)[ijk...]
    mask_ok(a) ? T(a) : T(_getfield(m, :beta)[ijk...])
end
mask_ok(a) = !ismissing(a) && !isnothing(a) && isfinite(a)

Base.showarg(io::IO, m::Mask, top=true) = print(io, typeof(m), top ? ", with $(count(!mask_ok, m.alpha)) from beta" : "")

#==== gradient stuff =====#

uncollect(dz::AbstractArray{T}, m::Mask) where {T<:Number} = mask_uncollect(dz, m.alpha)
function mask_uncollect(dz::AbstractArray{T}, alp) where {T<:Number}
    alpha = ifelse.(mask_ok.(alp), dz, zero(T))
    beta = ifelse.(.!mask_ok.(alp), dz, zero(T))
    (; alpha, beta)
end

# For structural gradient, the projector need not store the whole argument, only alpha.
# ChainRulesCore.ProjectTo(m::Mask) = ProjectTo{Mask}(; alpha = m.alpha, type=typeof(m))  # Val(typeof(m)) better
# function (p::ProjectTo{Mask})(dx::AbstractArray)
#     @info "projecting to Tangent{Mask}, II" typeof(dx)
#     Tangent{p.type}(; mask_uncollect(dx, p.alpha)...)
# end
# (p::ProjectTo{Mask})(dx::Tangent) = dx # Zygote makes Tangent{Any}, unfortunately

# But maybe the natural is very simple here?
naturalise(dx::TangentOrNT, ::Type{<:Mask}) = dx.alpha + dx.beta
restrict(dx::AbstractArray, ::Type{<:Mask}) = dx

# Let's use those:
ChainRulesCore.ProjectTo(m::Mask) = ProjectTo{Mask}()
(p::ProjectTo{Mask})(dx::AbstractArray) = dx
(p::ProjectTo{Mask})(dx::Tangent) = naturalise(dx, Mask)

"""
    PDiagMat(v)

This is like the one in [PDMats.jl](https://github.com/JuliaStats/PDMats.jl),
except it's an OddArray. Optimised `inv` (makes another) and `diag` (getter).
```
julia> PDiagMat([1,2,3])
3×3 PDiagMat{Float64, Vector{Float64}}:
 1.0  0.0  0.0
 0.0  2.0  0.0
 0.0  0.0  3.0

julia> NamedTuple(ans)
(dim = 3, diag = [1.0, 2.0, 3.0], inv_diag = [1.0, 0.5, 0.3333333333333333])
```
"""
struct PDiagMat{T<:Real, V<:AbstractVector} <: OddArray{T,2}
    dim::Int                    # matrix dimension
    diag::V                     # the vector of diagonal elements
    inv_diag::V                 # the element-wise inverse of diag
    PDiagMat(v::V, iv::V) where {V<:AbstractVector{T}} where {T<:Real} = new{T,V}(length(v), v, iv)
end
function PDiagMat(v::AbstractVector{<:Real})
    iv = inv.(v)
    PDiagMat(convert(typeof(iv), v), iv)
end

Base.size(m::PDiagMat) = (m.dim, m.dim)

Base.getindex(m::PDiagMat, i::Int, j::Int) = _getindex(m, i, j)
_getindex(m::PDiagMat{T}, i, j) where {T} = i==j ? _getfield(m, :diag)[i] : zero(T)

#==== optimisations of generic functions =====#

Base.inv(m::PDiagMat) = _inv(m)
function _inv(m::PDiagMat)
    _info("_inv(::PDiagMat)")
    PDiagMat(_getfield(m, :inv_diag), _getfield(m, :diag))
end

LinearAlgebra.diag(m::PDiagMat) = _diag(m)
_diag(m::PDiagMat) = _getfield(m, :diag)

function Base.:*(m::PDiagMat, n::PDiagMat)
    _info("*(::PDiagMat, ::PDiagMat)")
    PDiagMat(_getfield(m, :diag) .* _getfield(n, :diag), _getfield(m, :inv_diag) .* _getfield(n, :inv_diag))
end

#==== gradient stuff =====#

uncollect(dx::AbstractMatrix, x::PDiagMat) = uncollect(dx, typeof(x))
uncollect(dx::AbstractMatrix, ::Type{<:PDiagMat}) = (; dim=nothing, diag=diag(dx), inv_diag=nothing)

#= # opting out doesn't seem to work for inv, e.g.:

julia> gradient(x -> (x * inv(x))[1], PDiagMat([1,2,3]))
[ Info: _inv(::PDiagMat)
[ Info: *(::PDiagMat, ::PDiagMat)
┌ Info: projecting to Tangent{PDiagMat}
└   typeof(dx) = Matrix{Float64} (alias for Array{Float64, 2})
[ Info: generic _uncollect
ERROR: MethodError: no method matching (::OddArrays.var"#unconstruct#57"{PDiagMat{Float64, Vector{Float64}}})(::Tangent{Any, NamedTuple{(:dim, :diag, :inv_diag), Tuple{ZeroTangent, Vector{Float64}, ZeroTangent}}})

=#

# The projector Matrix -> Tangent need not store the argument at all.

ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(inv), ::PDiagMat)
ChainRulesCore.@opt_out ChainRulesCore.rrule(::typeof(*), ::PDiagMat, ::PDiagMat)

# function ChainRulesCore.rrule(::Type{<:PDiagMat}, v, iv)
#     y = PDiagMat(v, iv)
#     function unconstruct(dy::AbstractMatrix)
#         @info "adjoint constructor for PDiagMat" typeof(dy)
#         _, diag, inv_diag = uncollect(dy, y)
#         (NoTangent(), diag, inv_diag)
#     end
#     y, unconstruct
# end

#####
##### Types which are their own "natural" gradient represenation
#####

"""
    Full(x, size...)

Just like `Fill` from FillArrays, really, except it's an `OddArray`.
Has optimised methods for `sum`, `prod`, `*`, `+`.

Takes a keyword: `one=false` projects the gradient to another `Full`,
and `true` to a `OneElement`. Default is `ONEGRAD[]`, initially `false`.
```
julia> restrict([3 4 5], Full(pi,1,3))
1×3 Full{false, Float64, 2}:
 4.0  4.0  4.0

julia> restrict([3 4 5], Full(pi,1,3; one=true))
1×3 OneElement(::Int64):
 12  0  0
```
"""
struct Full{O,T,N} <: OddArray{T,N}
    value::T
    size::NTuple{N,Int}
    Full(val::T, size::NTuple{N,<:Integer}; one::Bool=ONEGRAD[]) where {T<:Number,N} = new{one,T,N}(val, size)
end
Full(val::Number, size::Integer...; kw...) = Full(val, size; kw...)

Base.size(f::Full) = f.size

Base.getindex(f::Full, ijk::Int...) = _getindex(f, ijk...)
_getindex(f::Full, ijk...) = _getfield(f, :value)

const ONEGRAD = Ref(false)
_onegrad(::Full{O}) where {O} = O

#==== optimisations of generic functions =====#

Base.sum(f::Full) = _sum(f)
_sum(f::Full) = f.value * length(f)
Base.prod(f::Full) = _prod(f)
_prod(f::Full) = f.value ^ length(f)

function Base.:*(f::Full, g::Full)
    _info("*(::Full, ::Full)")
    sz = (size(f)[1:end-1]..., size(g)[2:end]...)
    Full(f.value * g.value * size(f)[end], sz; one=_onegrad(f))
end
Base.:*(λ::Number, f::Full) = Full(λ * f.value, f.size; one=_onegrad(f))
Base.:*(f::Full, λ::Number) = Full(λ * f.value, f.size; one=_onegrad(f))
Base.:+(f::Full, g::Full) = f.size == g.size ? Full(f.value + g.value, f.size; one=_onegrad(f)) : throw("dimensionmismatch!")

#==== gradient stuff =====#

uncollect(dz::AbstractArray, f::Full) = size(dz) == f.size ? uncollect(dz, typeof(f)) : throw("dimensionmismatch!")
uncollect(dz::AbstractArray, ::Type{<:Full}) = (; value=sum(dz), size=nothing)

restrict(dz::AbstractArray, ::Type{<:Full{false}}) = Full(mean(dz), size(dz))
restrict(dz::AbstractArray, ::Type{<:Full{true}}) = OneElement(sum(dz), map(first, axes(dz)), axes(dz))
restrict(dz::AbstractArray, ::Type{<:Full}) = restrict(dz, Full{ONEGRAD[]})  # fall back to default?

naturalise(dz::TangentOrNT, f::Full{false}) = Full(dz.value / length(f), f.size)
naturalise(dz::TangentOrNT, f::Full{true}) = OneElement(dz.value, map(first, axes(f)), axes(f))
naturalise(dz::TangentOrNT, ::Type{<:Full}) = throw("naturalise needs the size of Full to produce")

ChainRulesCore.ProjectTo(f::Full) = ProjectTo{Full}(; size = f.size, one = _onegrad(f))
function (p::ProjectTo{Full})(dx::AbstractArray)
    size(dx) == p.size || throw("dimension mismatch!")
    @info "projecting to $(p.one ? "OneElement" : "Full")" typeof(dx)
    restrict(dx, Full{p.one})
end
function (p::ProjectTo{Full})(dx::Tangent)
    @info "projecting Tangent -> $(p.one ? "OneElement" : "Full")"
    Full(dx.value / prod(p.size), p.size; one = p.one)
end

function ChainRulesCore.rrule(::Type{<:Full}, val, size)
    function unfull(dy::AbstractArray)
        @info "adjoint constructor for Full" typeof(dy)
        (NoTangent(), uncollect(dy, Full)...)
    end
    Full(val, size), unfull
end

"""
    Alphabet(v)
    Alphabet(a,b,c,...)

This is a vector backed by a NamedTuple. Allows field access `x.a` etc,
and is mutable, `x[1] = 99`.
```
julia> abc = Alphabet(true,2,3.0)
3-element Alphabet{Float64, NamedTuple{(:a, :b, :c), Tuple{Bool, Int64, Float64}}}:
 1.0
 2.0
 3.0

julia> NamedTuple(abc)
(letters = (a = true, b = 2, c = 3.0),)

julia> abc.c = 300; abc  # is mutated

julia> gradient(x -> x.b, abc)
((letters = (a = nothing, b = 1.0, c = nothing),),)

julia> gradient(prod, abc)
┌ Info: projecting to Tangent{Alphabet}
└   typeof(dx) = Vector{Float64} (alias for Array{Float64, 1})
((letters = (a = 600.0, b = 300.0, c = 2.0),),)
```
"""
mutable struct Alphabet{T,NT} <: OddArray{T,1}
    letters::NT
    function Alphabet(abc::Number...)
        alpha = _alphabet(length(abc))
        type = mapreduce(typeof, promote_type, abc)
        data = NamedTuple{alpha}(abc)
        new{type, typeof(data)}(data)
    end
end
Alphabet(v::AbstractVector) = Alphabet(v...)
_alphabet(n) = ntuple(i -> Symbol('a'-1+i), n)
Base.parent(a::Alphabet) = _getfield(a, :letters)

Base.size(a::Alphabet) = (length(parent(a)),)

Base.getindex(a::Alphabet, i::Int) = _getindex(a, i)
_getindex(a::Alphabet{T}, i) where {T} = T(parent(a)[i])

function Base.setindex!(a::Alphabet{T,<:NamedTuple{A}}, val, i::Int) where {T,A}
    tup = Base.setindex(Tuple(parent(a)), T(val), i)
    a.letters = NamedTuple{A}(tup)
    a
end

Base.propertynames(a::Alphabet) = propertynames(parent(a))
Base.getproperty(a::Alphabet{T}, s::Symbol) where {T} = s === :letters ? parent(a) : T(getproperty(parent(a), s))

function Base.setproperty!(a::Alphabet{T}, s::Symbol, val::Number) where {T}
    s in propertynames(a) || throw("argument error? can't set $s")
    args = map(propertynames(a)) do n
        old = getfield(parent(a), n)
        n===s ? convert(typeof(old), val) : old 
    end
    setfield!(a, :letters, NamedTuple{propertynames(a)}(args))
end

#==== optimisations of generic functions =====#

Base.:+(a::Alphabet, b::Alphabet) = length(a) == length(b) ? 
    Alphabet((Tuple(parent(a)) .+ Tuple(parent(b)))...) : throw("dimensionmismatch!")

#==== gradient stuff =====#

function uncollect(dz::AbstractVector, a::Alphabet)
    length(dz) == length(a) || throw("dimensionmismatch!")
    uncollect(dz, typeof(a))
end
function uncollect(dz::AbstractVector, ::Type{<:Alphabet})
    letters = parent(Alphabet(dz))
    (; letters)
end

"""
    Range(α, ω, ℓ=5)

Like `range(α, ω, length=ℓ)` except `<: OddArray`.
"""
struct Range{T} <: OddArray{T,1}
    start::T
    stop::T
    len::Int
    Range(start::T1, stop::T2, len::Integer) where {T1<:Number, T2<:Number} = new{float(promote_type(T1,T2))}(start, stop, len)
end
Range(start, stop; length::Integer=5) = Range(start, stop, length)

Range(start::AbstractZero, stop::Number, length::Integer=5) = Range(false, stop, length)
Range(start::Number, stop::AbstractZero, length::Integer=5) = Range(start, false, length)

Base.size(r::Range) = (_getfield(r, :len),)

Base.getindex(r::Range, i::Int) = _getindex(r, i)
_getindex(r::Range, i) = _getfield(r, :start) + _step(r) * (i - 1)

Base.step(r::Range) = _step(r)
_step(r) = (_getfield(r, :stop) - _getfield(r, :start))/(_getfield(r, :len) - 1)

Base.first(r::Range) = _getfield(r, :start)
Base.last(r::Range) = _getfield(r, :stop)

Base.showarg(io::IO, r::Range, top=true) = print(io, "Range{", eltype(r), "}", top ? ", with step = $(step(r))" : "")

#==== optimisations of generic functions =====#

Base.:+(r::Range, s::Range) = r.len == s.len ? Range(_getfield(r, :start) + _getfield(s, :start), _getfield(r, :stop) + _getfield(s, :stop), _getfield(r, :len)) : throw("dimension mismatch")
Base.:-(r::Range, s::Range) = r.len == s.len ? Range(_getfield(r, :start) - _getfield(s, :start), _getfield(r, :stop) - _getfield(s, :stop), _getfield(r, :len)) : throw("dimension mismatch")
Base.:-(r::Range) = Range(-_getfield(r, :start), -_getfield(r, :stop), _getfield(r, :len))
Base.:*(λ::Number, r::Range) = Range(λ * _getfield(r, :start), λ * _getfield(r, :stop), _getfield(r, :len))
Base.:*(r::Range, λ::Number) = Range(λ * _getfield(r, :start), λ * _getfield(r, :stop), _getfield(r, :len))

#==== gradient stuff =====#

function uncollect(dz::AbstractVector, r::Range)
    length(dz) == r.len || throw("dimensionmismatch!")
    uncollect(dz, typeof(r))
end
function uncollect(dz::AbstractVector, ::Type{<:Range})
    start = dot(dz, LinRange(1,0,length(dz)))
    stop = dot(dz, LinRange(0,1,length(dz)))
    len = nothing
    (; start, stop, len)
end

# restrict(dz::AbstractVector, r::Range) = length(dz) == r.len ? restrict(dz, typeof(r)) : throw("dimensionmismatch!")
# function restrict(dz::AbstractVector, ::Type{<:Range})
#     μ = mean(dz)
#     δ = (dz[end] - dz[begin])/2
#     Range(μ - δ, μ + δ, length(dz))
# end

function naturalise(dz::TangentOrNT, r::Range)
    α, β = dz.start, dz.stop
    ℓ = length(r)
    start = (4(ℓ^2)*(α+β) - 2ℓ*(α+4β) + 6β) / (ℓ*(ℓ+1))
    stop = (- 2(ℓ^2)*(α+β) + 2ℓ*(2α+5β) - 6β) / (ℓ*(ℓ+1))
    Range(start, stop, ℓ)
end
naturalise(dz::TangentOrNT, ::Type{<:Range}) = throw("naturalise needs the length of Range to produce")

ChainRulesCore.ProjectTo(r::Range) = ProjectTo{Range}(; len = r.len)
(p::ProjectTo{Range})(dx::Range) = dx
function (p::ProjectTo{Range})(dx::AbstractVector)
    length(dx) == p.len || throw("dimensionmismatch!")
    @info "projecting to Range" typeof(dx)
    restrict(dx, Range)
end
# function (p::ProjectTo{Range})(dx::Tangent)
#     @info "projecting Tangent -> Range"
#     naturalise(dx, Range(0,0,p.len))
# end

# function ChainRulesCore.rrule(::Type{<:Range}, start, stop, len)
#     # function unrange(dy::Range)
#     #     @info "adjoint constructor for Range"
#     #     @show dy.start
#     #     @show dy.stop
#     #     (NoTangent(), dy.start, dy.stop, ZeroTangent())
#     # end
#     function unrange(dy::AbstractVector)
#         @info "adjoint constructor for Range" typeof(dy)
#         ℓ = length(dy)
#         μ = mean(dy)
#         δ = sum(Base.splat(-), zip(dy, @view dy[2:end]))/2
#         m = dot(dy, LinRange(1,0,ℓ))
#         p = dot(dy, LinRange(0,1,ℓ))
#         @show μ δ m p 
#         (NoTangent(), p+m, p+m, ZeroTangent())
#     end
#     Range(start, stop, len), unrange
# end


#####
##### some LinearAlgebra types are honorary OddArrays
#####

function Base.NamedTuple(x::Union{Diagonal, Symmetric, OneElement})
    F = fieldnames(typeof(x))
    NamedTuple{F}(map(f -> getfield(x, f), F))
end

#==== Diagonal =====#

_getindex(d::Diagonal, i::Int, j::Int) = i==j ? _getfield(d, :diag)[i] : zero(eltype(d))

uncollect(dx::AbstractMatrix, x::Diagonal) = uncollect(dx, typeof(x))
uncollect(dx::AbstractMatrix, ::Type{<:Diagonal}) = (; diag = diag(dx))

"""
    restrict(dx::Array, x::OddArray) -> OddArray
    restrict(dx::Array, ::Type{OddArray}) -> OddArray

This projects from the full space onto a "natural" representation,
when possible. Always accepts `x`, sometimes accepts just the type.
"""
restrict(dx::AbstractMatrix, x::Diagonal) = Diagonal(dx)
restrict(dx::AbstractMatrix, ::Type{<:Diagonal}) = Diagonal(dx)

"""
    naturalise(dx::Tangent, x::OddArray) -> OddArray
    naturalise(dx::Tangent, ::Type{OddArray}) -> OddArray

This maps a "structural" representation to a "natural" one.
"""
naturalise(dx::TangentOrNT, x::Diagonal) = Diagonal(dx.diag)
naturalise(dx::TangentOrNT, ::Type{<:Diagonal}) = Diagonal(dx.diag)

#==== Symmetric =====#

function _getindex(s::Symmetric, i::Int, j::Int)
    data = _getfield(s, :data)
    if s.uplo === 'U'
        i>j ? data[j,i] : data[i,j]
    else
        @assert s.uplo === 'L'
        i<j ? data[j,i] : data[i,j]
    end
end

function uncollect(dx::AbstractMatrix, x::Symmetric)
    if x.uplo === 'U'
        data = UpperTriangular(dx) + transpose(UnitLowerTriangular(dx)) - I
    else
        data = LowerTriangular(dx) + transpose(UnitUpperTriangular(dx)) - I
    end
    (; data=data, uplo=nothing)
end
uncollect(dx::AbstractMatrix, ::Type{<:Symmetric}) = throw("uncollect needs to see uplo field of Symmetric")

restrict(dx::AbstractMatrix, x::Symmetric) = restrict(dx, typeof(x))
restrict(dx::AbstractMatrix, ::Type{<:Symmetric}) = Symmetric((dx .+ transpose(dx))./2)
restrict(dx::Symmetric, ::Type{<:Symmetric}) = dx


#####
##### Piracy, for tests
#####

function Base.isapprox(x::NamedTuple, y::NamedTuple; kw...)
    Set(propertynames(x)) == Set(propertynames(y)) || return false
    for n in propertynames(x)
        x_n = getfield(x, n)
        isnothing(x_n) && continue
        y_n = getfield(y, n)
        isnothing(y_n) && continue
        isapprox(x_n, y_n; kw...) || return false
    end
    return true
end

end # module
