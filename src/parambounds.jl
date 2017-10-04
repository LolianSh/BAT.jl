# This file is a part of BAT.jl, licensed under the MIT License (MIT).


doc"""
    nparams(X::Union{AbstractParamBounds,MCMCIterator,...})

Get the number of parameters of `X`.
"""
function nparams end
export nparams



export AbstractParamBounds

abstract type AbstractParamBounds{T<:Real} end

Base.eltype(b::AbstractParamBounds{T}) where T = T

Base.rand(rng::AbstractRNG, bounds::AbstractParamBounds) =
    rand!(rng, bounds, Vector{float(eltype(bounds))}(nparams(bounds)))

Base.rand(rng::AbstractRNG, bounds::AbstractParamBounds, n::Integer) =
    rand!(rng, bounds, Matrix{float(eltype(bounds))}(nparams(bounds), n))



@inline oob{T<:AbstractFloat}(::Type{T}) = T(NaN)
@inline oob{T<:Integer}(::Type{T}) = typemax(T)
@inline oob(x::Real) = oob(typeof(x))
oob(xs::AbstractArray) = fill!(similar(xs), oob(eltype(xs)))

@inline isoob(x::AbstractFloat) = isnan(x)
@inline isoob(x::Integer) = x == oob(x)
isoob(xs::AbstractArray) = any(isoob, xs)


@enum BoundsType hard_bounds=1 cyclic_bounds=2 reflective_bounds=3
export BoundsType
export hard_bounds, cyclic_bounds, reflective_bounds


@inline float_iseven(n::T) where {T<:AbstractFloat} = (n - T(2) * floor((n + T(0.5)) * T(0.5))) < T(0.5)

doc"""
    apply_bounds(x::<:Real, lo::<:Real, hi::<:Real, boundary_type::BoundsType) 

Set low bound `lo` and high bound `hi` for Parameter `x`
`boundary_type` may be `hard_bounds`, `cyclic_bounds` or `reflective_bounds`.
"""
@inline function apply_bounds(x::X, lo::L, hi::H, boundary_type::BoundsType, oobval = oob(x)) where {X<:Real,L<:Real,H<:Real}
    T = float(promote_type(X, L, H))

    offs = ifelse(x < lo, lo - x, x - hi)
    hi_lo = hi - lo
    nwrapped = floor(offs / (hi_lo))
    even_nwrapped = float_iseven(nwrapped)
    wrapped_offs = muladd(-nwrapped, (hi_lo), offs)

    hb = (boundary_type == hard_bounds)
    rb = (boundary_type == reflective_bounds)

    ifelse(
        lo <= x <= hi,
        convert(T, x),
        ifelse(
            hb,
            convert(T, oobval),
            ifelse(
                (x < lo && (!rb || rb && !even_nwrapped)) || (x > lo && rb && even_nwrapped),
                convert(T, hi - wrapped_offs),
                convert(T, lo + wrapped_offs)
            )
        )
    )
end

doc"""
    apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType)

Instead of `lo` and `hi` an `interval` can be used.
"""
@inline apply_bounds(x::Real, interval::ClosedInterval, boundary_type::BoundsType, oobval = oob(x)) =
    apply_bounds(x, minimum(interval), maximum(interval), boundary_type, oobval)



export UnboundedParams

struct UnboundedParams{T<:Real} <: AbstractParamBounds{T}
    ndims::Int
end

Base.in(params::AbstractVector, bounds::UnboundedParams) = true
Base.in(params::AbstractMatrix, bounds::UnboundedParams, i::Integer) = true

nparams(b::UnboundedParams) = b.ndims


doc"""
    apply_bounds!(params::AbstractVector, bounds::UnboundedParams) 

For Parameters without bounds use `bounds` of type `UnboundedParams`
"""
apply_bounds!(params::AbstractVector, bounds::UnboundedParams) = params



export ParamVolumeBounds

abstract type ParamVolumeBounds{T<:Real, V<:SpatialVolume{T}} <: AbstractParamBounds{T} end

Base.in(params::AbstractVector, bounds::ParamVolumeBounds) = in(params, bounds.vol)
Base.in(params::AbstractMatrix, bounds::ParamVolumeBounds, j::Integer) = in(params, bounds.vol, j)

Base.rand!(rng::AbstractRNG, bounds::ParamVolumeBounds, x::StridedVecOrMat{<:Real}) = rand!(rng, bounds.vol, x)

nparams(b::ParamVolumeBounds) = ndims(b.vol)



export HyperRectBounds

struct HyperRectBounds{T<:Real} <: ParamVolumeBounds{T, HyperRectVolume{T}}
    vol::HyperRectVolume{T}
    bt::Vector{BoundsType}

    function HyperRectBounds{T}(vol::HyperRectVolume{T}, bt::Vector{BoundsType}) where {T<:Real}
        indices(bt) != (1:ndims(vol),) && throw(ArgumentError("bt must have indices (1:ndims(vol),)"))
        new{T}(vol, bt)
    end
end


HyperRectBounds{T<:Real}(vol::HyperRectVolume{T}, bt::AbstractVector{BoundsType}) = HyperRectBounds{T}(vol, bt)
HyperRectBounds{T<:Real}(lo::AbstractVector{T}, hi::AbstractVector{T}, bt::AbstractVector{BoundsType}) = HyperRectBounds(HyperRectVolume(lo, hi), bt)
HyperRectBounds{T<:Real}(lo::AbstractVector{T}, hi::AbstractVector{T}, bt::BoundsType) = HyperRectBounds(lo, hi, fill(bt, size(lo, 1)))

function apply_bounds!(params::AbstractVecOrMat, bounds::HyperRectBounds, setoob = true)
    if setoob
        params .= apply_bounds.(params, bounds.vol.lo, bounds.vol.hi, bounds.bt)
    else
        params .= apply_bounds.(params, bounds.vol.lo, bounds.vol.hi, bounds.bt, params)
    end
end
