# This file is a part of BAT.jl, licensed under the MIT License (MIT).


abstract type AbstractMCMCState end


sample_available(state::AbstractMCMCState) = sample_available(state, Val(:complete))

current_sample(state::AbstractMCMCState) = current_sample(state, Val(:complete))



abstract type MCMCAlgorithm{S<:AbstractMCMCState} end


abstract type AbstractMCMCSample end
export AbstractMCMCSample



mutable struct MCMCSample{
    P<:Real,
    T<:Real,
    W<:Real
} <: AbstractMCMCSample
    params::Vector{P}
    log_value::T
    weight::W
end

export MCMCSample


Base.length(s::MCMCSample) = length(s.params)

Base.similar(s::MCMCSample{P,T,W}) where {P,T,W} =
    MCMCSample{P,T,W}(oob(s.params), convert(T, NaN), zero(W))

import Base.==
==(A::MCMCSample, B::MCMCSample) =
    A.params == B.params && A.log_value == B.log_value && A.weight == B.weight


function Base.copy!(dest::MCMCSample, src::MCMCSample) 
    copy!(dest.params, src.params)
    dest.log_value = src.log_value
    dest.weight = src.weight
    dest
end


nparams(s::MCMCSample) = length(s)



struct MCMCChainInfo
    id::Int
    cycle::Int
    tuned::Bool
    converged::Bool
end

export MCMCChainInfo

MCMCChainInfo(id::Int, cycle::Int = 0) = MCMCChainInfo(id, cycle, false, false)


next_cycle(info::MCMCChainInfo) =
    MCMCChainInfo(info.id, info.cycle + 1, info.tuned, info.converged)

set_tuned(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, value, info.converged)

set_converged(info::MCMCChainInfo, value::Bool) =
    MCMCChainInfo(info.id, info.cycle, info.tuned, value)



mutable struct MCMCChain{
    A<:MCMCAlgorithm,
    T<:AbstractTargetSubject,
    S<:AbstractMCMCState
}
    algorithm::A
    target::T
    state::S
    info::MCMCChainInfo
end

export MCMCChain


nparams(chain::MCMCChain) = nparams(chain.target)

sample_available(chain::MCMCChain, status::Val = Val(:complete)) = sample_available(chain.state, status)

current_sample(chain::MCMCChain, status::Val = Val(:complete)) = current_sample(chain.state, status)

current_sampleno(chain::MCMCChain) = current_sampleno(chain.state)



"""
    AbstractMCMCCallback <: Function

Subtypes (here, `X`) must support

    (::X)(level::Integer, chain::MCMCChain) => nothing
    (::X)(level::Integer, tuner::AbstractMCMCTuner) => nothing

to be compabtible with `mcmc_iterate!`, `mcmc_tune_burnin!`, etc.
"""
abstract type AbstractMCMCCallback <: Function end
export AbstractMCMCCallback



"""
    mcmc_callback(
        output::Union{Any,AbstractMCMCStats,MCMCSampleVector,...},
        max_level::Integer = 1
    )::AbstractMCMCCallback

Creates a callback function/object compatible with `mcmc_iterate!`,
`mcmc_tune_burnin!`, etc., that will fill `output` with samples
generated by the chain. Depending on the output, `max_level` may be ignored.

    mcmc_callback(fs::Tuple)::AbstractMCMCCallback

Creates a callback that broadcasts it's arguments over all functions in
the tuple `fs`.

    mcmc_callback(f::Any) = f

This variant assumes that `f` is already a compabible callback function.
"""
function mcmc_callback end
export mcmc_callback

mcmc_callback(f::Any) = f

mcmc_callback(cb::AbstractMCMCCallback) = cb



struct MCMCMultiCallback{FT<:Tuple} <: AbstractMCMCCallback
    funcs::FT
end


function (cb::MCMCMultiCallback)(level::Integer, chain::MCMCChain)
    map(f -> f(level, chain), cb.funcs)
    nothing
end

function mcmc_callback(funcs::Tuple)
    cb_funcs =  map(f -> mcmc_callback(f), funcs)
    MCMCMultiCallback(cb_funcs)
end



struct MCMCPushCallback{T} <: AbstractMCMCCallback
    target::T
    max_level::Int
end

MCMCPushCallback(target) = MCMCPushCallback(target, 1)

function (cb::MCMCPushCallback)(level::Integer, chain::MCMCChain)
    if (level <= cb.max_level)
        push!(cb.target, chain)
    end
    nothing
end

