"""
    OddArrays

This defines some array types, with more complicated relation
between storage and elements than the ones in LinearAlgebra.

Call `_getindex` or `_collect` to bypass the Base functions,
and hence their gradient definitions.
"""
module OddArrays
export OddArray, Rotation, LogRange, Vandermonde, Outer, Full, Mask,
    decollect, _decollect, _getindex, _collect, _det, _prod, _sum

using LinearAlgebra, ChainRulesCore, Zygote
export Diagonal, ProjectTo, Tangent, gradient, pullback

abstract type OddArray{T,N} <: AbstractArray{T,N} end

Base.collect(x::OddArray) = _collect(x)
_collect(x::OddArray) = reshape([_getindex(x, I.I...) for I in CartesianIndices(size(x))], size(x))
ChainRulesCore.@non_differentiable CartesianIndices(::Tuple)

"""
    decollect(dx::Array, x::OddArray) -> NamedTuple

This turns a natural into a structural representation.
Called by `ProjectTo{OddArray}(::Array)` by default.

Fallback implementation is `_decollect`, which calls `_collect`
using Zygote, and discards its forward pass.
"""
decollect(dx::AbstractArray, x::OddArray) = _decollect(dx, x)
function _decollect(dx::AbstractArray, x::OddArray)
    _, back = Zygote.pullback(_collect, x)
    @info "generic _decollect"
    back(dx)[1]
end

ChainRulesCore.ProjectTo(x::OddArray) = ProjectTo{OddArray}(; x = x)
function (p::ProjectTo{OddArray})(dx::AbstractArray)
    @info "projecting to Tangent{$(typeof(p.x).name.name)}" typeof(dx)
    Tangent{typeof(p.x)}(; decollect(dx, p.x)...)
end

(p::ProjectTo{OddArray})(dx::Tangent) = dx # Zygote makes Tangent{Any}, unfortunately

"""
    Rotation(θ)

This is a 2x2 rotation matrix, parameterised by the angle.
Has an optimised method for multiplication with a vector, via `mul!`,
and multiplication of two rotations, `*`.
"""
struct Rotation{T} <: OddArray{T,2}
    theta::T
    Rotation(theta::T) where {T<:Number} = new{float(T)}(float(theta))
end

Base.size(r::Rotation) = (2,2)

Base.getindex(r::Rotation, i::Int, j::Int) = _getindex(r, i, j)
function _getindex(r::Rotation, i, j)
    checkbounds(r, i, j)
    if i==j
        return cos(r.theta)
    else
        s = sin(r.theta)
        return i>j ? s : -s
    end
end

function Base.:*(r::Rotation, s::Rotation)
    _info("*(::Rotation, ::Rotation)")
    Rotation(r.theta + s.theta)
end

_info(s::String) = @info s
Zygote.@nograd _info

function LinearAlgebra.mul!(y::AbstractVector, r::Rotation, x::AbstractVector)
    _info("mul!(_, ::Rotation, _)")
    s, c = sincos(r.theta)
    x1, x2 = x
    y[1] = c * x1 - s * x2
    y[2] = s * x1 + c * x2
    y
end

function decollect(dx::AbstractMatrix, r::Rotation)
    s, c = sincos(r.theta)
    theta = - s * dx[1,1] - c * dx[1,2] + c * dx[2,1] - s * dx[2,2]
    @info "decollect(_, ::Rotation)" theta
    (; theta)
end

"""
    LogRange(α, ω, ℓ=5)

Like `range(α, ω, length=ℓ)` except geometrically not linearly spaced,
i.e. constant ratio not difference.
Has an optimised method for `prod`, exported as `_prod`.
"""
struct LogRange{T} <: OddArray{T,1}
    start::T
    stop::T
    len::Int
    LogRange(start::T1, stop::T2, len::Integer) where {T1<:Number, T2<:Number} = new{float(promote_type(T1,T2))}(start, stop, len)
end
Base.size(l::LogRange) = (l.len,)
Base.getindex(l::LogRange, i::Int) = _getindex(l, i)
_getindex(l::LogRange, i) = l.start * (l.stop/l.start)^((i-1)/(l.len-1))

Base.prod(l::LogRange) = _prod(l)
_prod(l::LogRange) = (l.start * l.stop)^(l.len/2)

"""
    Vandermonde(x, s=1)

Defines an NxN matrix given a vector of N coefficients,
whose values are `hcat((x .^ i for i in 0:N-1)...) .* s`.
Has an optimised method for `det`, exported as `_det`.
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

Lazy outer product matrix. If `x, y` are vectors then this has 2N parameters for N^2.
But other shapes are accepted, which allow this to have more than N^2 parameters.
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

decollect(dz::AbstractMatrix, o::Outer{<:Any,<:AbstractVector,<:AbstractVector}) =
    (; x = dz * o.y, y = dz' * o.x, size=nothing)
decollect(dz::AbstractMatrix, o::Outer{<:Any,<:AbstractMatrix,<:Number}) =
    (; x=dz .* o.y, y=dot(dz, o.x), size=nothing)  # wrong for complex numbers?
decollect(dz::AbstractMatrix, o::Outer{<:Any,<:Number,<:AbstractMatrix}) =
    (; x=dot(dz, o.y'), y=dz' .* o.x, size=nothing)  # wrong for complex numbers?

"""
    Full(x, size...)

Just like `Fill` from FillArrays, really.
Has optimised methods for `sum`, `prod`, `*`.
"""
struct Full{T,N} <: OddArray{T,N}
    value::T
    size::NTuple{N,Int}
    Full(x::T, size::NTuple{N,<:Integer}) where {T<:Number,N} = new{T,N}(x, size)
end
Full(x::Number, size::Integer...) = Full(x, size)

Base.size(f::Full) = f.size

Base.getindex(f::Full, ijk::Int...) = _getindex(f, ijk...)
_getindex(f::Full, ijk...) = f.value

Base.sum(f::Full) = _sum(f)
_sum(f::Full) = f.value * length(f)
Base.prod(f::Full) = _prod(f)
_prod(f::Full) = f.value ^ length(f)

function Base.:*(f::Full, g::Full)
    _info("*(::Full, ::Full)")
    sz = (size(f)[1:end-1]..., size(g)[2:end]...)
    Full(f.value * g.value * size(f)[end], sz)
end

decollect(dz::AbstractArray, f::Full) = (; value=sum(dz), size=nothing)

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

Base.size(m::Mask) = size(m.alpha)

Base.getindex(m::Mask, ijk::Int...) = _getindex(m, ijk...)
function _getindex(m::Mask{T}, ijk...) where {T}
    a = m.alpha[ijk...]
    mask_ok(a) ? T(a) : T(m.beta[ijk...])
end
mask_ok(a) = !ismissing(a) && !isnothing(a) && isfinite(a)

decollect(dz::AbstractArray{T}, m::Mask) where {T<:Number} = mask_decollect(dz, m.alpha)
function mask_decollect(dz::AbstractArray{T}, alp) where {T<:Number}
    alpha = ifelse.(mask_ok.(alp), dz, zero(T))
    beta = ifelse.(.!mask_ok.(alp), dz, zero(T))
    (; alpha, beta)
end

# In fact the projector need not store the whole argument, only alpha:
ChainRulesCore.ProjectTo(m::Mask) = ProjectTo{Mask}(; alpha = m.alpha, type=typeof(m))  # Val(typeof(m)) better
function (p::ProjectTo{Mask})(dx::AbstractArray)
    @info "projecting to Tangent{Mask}, II" typeof(dx)
    Tangent{p.type}(; mask_decollect(dx, p.alpha)...)
end
(p::ProjectTo{Mask})(dx::Tangent) = dx # Zygote makes Tangent{Any}, unfortunately

end # module
