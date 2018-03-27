# This file is a part of BAT.jl, licensed under the MIT License (MIT).


mutable struct GeneralizedMHState{
    Q<:AbstractProposalDist,
    SV<:DensitySampleVector
} <: AbstractMCMCState
    pdist::Q
    samples::SV  # First element is the current sample in the chain
    accepted::Vector{Bool}
    eff_naccept::Double{Float64}
    nsamples::Int64
    nsteps::Int64
end

function GeneralizedMHState(
    pdist::AbstractProposalDist,
    current_sample::DensitySample{P,T,W},
    nproposals::Int
) where {P,T,W}
    npar = nparams(current_sample)

    params = ElasticArray{T}(npar, nproposals + 1)
    fill!(params, zero(T))
    params[:, 1] = current_sample.params

    log_value = Vector{T}(nproposals + 1)
    fill!(log_value, NaN)
    log_value[1] = current_sample.log_value

    weight = Vector{W}(nproposals + 1)
    fill!(weight, zero(W))
    weight[1] = current_sample.weight

    samples = DensitySampleVector(params, log_value, weight)
    accepted = fill(false, )

    eff_naccept = 0
    nsamples = 0
    nsteps = 0

    GeneralizedMHState(
        pdist,
        samples,
        eff_naccept,
        nsamples,
        nsteps
    )
end


nparams(state::GeneralizedMHState) = nparams(state.pdist)

nsteps(state::GeneralizedMHState) = state.nsteps

nsamples(state::GeneralizedMHState) = state.nsamples

eff_acceptance_ratio(state::GeneralizedMHState) = Float64(state.eff_naccept / state.nsteps)


function next_cycle!(state::GeneralizedMHState)
    state.samples.weight[1] = one(eltype(state.samples.weight))
    state.eff_naccept = zero(state.eff_naccept)
    state.nsamples = zero(state.nsamples)
    state.nsteps = zero(state.nsteps)
    state
end


function MCMCBasicStats(state::GeneralizedMHState)
    L = promote_type(eltype(state.samples.log_value), Float64)
    P = promote_type(eltype(state.samples.params), Float64)
    m = nparams(state)
    MCMCBasicStats{L, P}(m)
end


function nsamples_available(state::GeneralizedMHState; nonzero_weight::Bool = false)
    # ignore nonzero_weight for now
    # ToDo: Handle this properly
    length(state.samples)
end


function Base.append!(xs::DensitySampleVector, state::GeneralizedMHState)
    if nsamples_available(state) > 0
        new_samples = view(state.samples, (firstindex(state.samples) + 1):lastindex(state.samples))  # Memory allocation!
        append!(xs, new_samples)
    end
    xs
end



function current_sampleno(state::GeneralizedMHState)
    state.nsamples + 1
end

function current_stepno(state::GeneralizedMHState)
    state.nsteps
end



struct GeneralizedMetropolisHastings{
    Q<:ProposalDistSpec,
} <: MCMCAlgorithm{GeneralizedMHState}
    q::Q
    nproposals::Int
end

export GeneralizedMetropolisHastings

GeneralizedMetropolisHastings(q::ProposalDistSpec = MvTDistProposalSpec()) =
    GeneralizedMetropolisHastings(q, 10)


mcmc_compatible(::GeneralizedMetropolisHastings, ::AbstractProposalDist, ::NoParamBounds) = true

mcmc_compatible(::GeneralizedMetropolisHastings, pdist::AbstractProposalDist, bounds::HyperRectBounds) =
    issymmetric(pdist) || all(x -> x == hard_bounds, bounds.bt)

sample_weight_type(::Type{GeneralizedMetropolisHastings}) = Float64  # ToDo: Allow for other floating point types


function MCMCIterator(
    algorithm::GeneralizedMetropolisHastings,
    likelihood::AbstractDensity,
    prior::AbstractDensity,
    id::Int64,
    rng::AbstractRNG,
    initial_params::AbstractVector{P} = Vector{P}(),
    exec_context::ExecContext = ExecContext(),
) where {P<:Real}
    target = likelihood * prior

    cycle = zero(Int)
    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 0))

    params_vec = Vector{P}(nparams(target))
    if isempty(initial_params)
        rand_initial_params!(rng, algorithm, prior, params_vec)
    else
        params_vec .= initial_params
    end

    !(params_vec in param_bounds(target)) && throw(ArgumentError("Initial parameter(s) out of bounds"))

    reset_rng_counters!(rng, MCMCSampleID(id, cycle, 1))

    m = length(params_vec)

    log_value = density_logval(target, params_vec, exec_context)
    L = typeof(log_value)
    W = sample_weight_type(typeof(algorithm))

    current_sample = DensitySample(
        params_vec,
        log_value,
        one(W)
    )

    state = GeneralizedMHState(
        target,
        current_sample,
        algorithm.nproposals
    )

    chain = MCMCIterator(
        algorithm,
        target,
        state,
        rng,
        id,
        cycle,
        false,
        false
    )

    chain
end


function mcmc_step!(
    callback::AbstractMCMCCallback,
    chain::MCMCIterator{<:MCMCAlgorithm{GeneralizedMHState}},
    exec_context::ExecContext,
    ll::LogLevel
)
    state = chain.state
    algorithm = chain.algorithm

    if !mcmc_compatible(algorithm, chain.state.pdist, param_bounds(chain.target))
        error("Implementation of algorithm $algorithm does not support current parameter bounds with current proposal distribution")
    end

    state.nsteps += 1
    reset_rng_counters!(chain.rng, MCMCSampleID(chain, 1))

    rng = chain.rng
    target = chain.target
    pdist = state.pdist

    current_sample = state.current_sample
    proposed_sample = state.proposed_sample

    current_params = current_sample.params
    proposed_params = proposed_sample.params

    current_log_value = current_sample.log_value
    T = typeof(current_log_value)

    all_samples = state.samples
    all_params = all_samples.params
    all_weights = all_samples.weights
    all_logdensity_values = all_samples.log_value
    is_inbounds = BitVector(size(all_params, 2))  # Memory allocation!
    P_T1 = Vector{eltype(all_logdensity_values)}(size(all_params, 2))  # Memory allocation!
    #!! multi_propose!(rng, pdist, target, all_params, is_inbounds::BitVector)
    #!! multipropT1!(rng, pdist, target, all_params, all_logdensity_values, is_inbounds::BitVector, P_T1)
    #!! multipropT2!(P_T2::AbstractVector{<:AbstractFloat}, P_T1::AbstractVector{<:AbstractFloat})
    # ToDo: store P_T values in all_weights
    #!! accepted_sample = multiprop_select(...)
    # ToDo: swap all_samples[1] and all_samples[accepted_sample]

    # ToDo: preserve information about accepted sample in state of chain

    # ToDo: Granular callback
    callback(1, chain)

    chain
end
