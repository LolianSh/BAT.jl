# This file is a part of BAT.jl, licensed under the MIT License (MIT).


struct DirectSampling <: MCMCAlgorithm{AcceptRejectState} end
export DirectSampling


# ToDo: Specialized version of rand_initial_params for DirectSampling:
#
#     rand_initial_params!(rng::AbstractRNG, algorithm::DirectSampling, target::TargetSubject, x::StridedVecOrMat{<:Real}) = ...


AbstractMCMCTunerConfig(algorithm::DirectSampling) = NoOpTunerConfig()


sample_weight_type(::Type{DirectSampling}) = Int


function AcceptRejectState(
    algorithm::DirectSampling,
    target::AbstractTargetSubject, #{<:MvDistTargetDensity},
    current_sample::DensitySample{P,T,W}
) where {P,T,W}
    AcceptRejectState(
        GenericProposalDist(target.tdensity.d),
        current_sample
    )
end


function mcmc_propose_accept_reject!(
    callback::Function,
    chain::MCMCIterator{<:DirectSampling},
    exec_context::ExecContext
)
    state = chain.state
    target = chain.target

    proposed_sample = state.proposed_sample
    proposed_params = proposed_sample.params

    # Propose new parameters:
    rand!(chain.rng, state.pdist.s, proposed_params)

    # Accept iff in bounds:
    if proposed_params in target.bounds
        proposed_sample.log_value = target_logval(target.tdensity, proposed_params, exec_context)

        state.proposed_sample.weight = 1
        @assert state.current_sample.weight == 1
        state.proposal_accepted = true
        state.nsamples += 1
        callback(1, chain)
    else
        proposed_sample.log_value = -Inf

        state.current_nreject += 1
        callback(2, chain)
    end

    chain
end
