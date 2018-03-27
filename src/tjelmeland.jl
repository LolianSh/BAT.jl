# This file is a part of BAT.jl, licensed under the MIT License (MIT).


# Note: Matrices stored with swapped indices to Tjelmeland's paper,
# due to Julia's array memory layout.



# Calculate min(u) for Tjelmeland T2
function _tjl_min_u(P::AbstractMatrix{<:AbstractFloat})
    idxs = indices(P, 1)
    @assert idxs == indices(P, 2)
    length(idxs) < 2 && throw(ArgumentError("Size of matrix P must be at least 2x2"))
    u = typemax(eltype(P))
    @inbounds for k in idxs
        s = sum_first_dim(P, k)
        u = min(u, s / (s - P[k, k]))
    end
    u
end


function _tjl_update_sub_P!(P::AbstractMatrix{<:AbstractFloat}, u::AbstractFloat)
    idxs = indices(P, 1)
    @assert idxs == indices(P, 2)
    @inbounds for k in idxs
        s = sum_first_dim(P, k)
        P[k, k] = 1 - u * (s - P[k, k]) - (1 - s)
    end
    P .*= u
    inv_u = inv(u)
    for k in idxs
        P[k, k] *= inv_u
    end
    P
end


function _tjl_update_row!(row::Vector, submat_row::Vector, index::Vector, indexnew::Vector)
    for (i,a) in zip(index, indexnew)
        row[i] = submat_row[a]
    end
end


function _tjl_update_selidxs!(selidxs::AbstractVector{<:Integer}, P::AbstractMatrix{<:AbstractFloat})
    j = first(eachindex(selidxs))
    while j in eachindex(selidxs)
        if P[j, j] ≈ 0
            splice!(selidxs, j)
        else
            j += 1
        end
    end
end

doc"""
    multipropT2(P_T2::AbstractVector, P_T1::AbstractVector)

Compute the transition probabilities T2 from Tjelmeland (2002).

`P_transition` contains the T1 transition probabilities to the proposed
states, the first entry refers to the current state ( probability to stay in
place).

"""
function multipropT2!(P_T2::AbstractVector{<:AbstractFloat}, P_T1::AbstractVector{<:AbstractFloat})
    idxs = eachindex(P_T2)
    idxs != eachindex(P_T1) && throw(ArgumentError("P_T2 and P_T1 must have the same indices"))
    any(x -> x < 0, P_T1) && throw(ArgumentError("All values in P_T1 must be positive"))
    !(sum(P_T1) ≈ 1) && throw(ArgumentError("Sum of P_T1 must be one"))

    # Construct initial P matrix
    P = repeat(P_T1, outer=(1, length(idxs)))  # Memory allocation!

    selidxs = collect(idxs)  # Memory allocation!

    done = false
    while !done
        _tjl_update_selidxs!(selidxs, P)
        P_sub = view(P, selidxs, selidxs)  # Memory allocation!
        if (length(selidxs) < 2) || P[1, 1] ≈ 0
            @inbounds for l in idxs
                P_T2[l] = P[l, 1]
            end
            done = true
        else
            u = _tjl_min_u(P_sub)
            _tjl_update_sub_P!(P_sub, u)
        end
    end

    P_T2
end

multipropT2(P_T1::AbstractVector{<:AbstractFloat}) = multipropT2!(similar(P_T1), P_T1)

<<<<<<< Updated upstream
<<<<<<< Updated upstream
<<<<<<< Updated upstream
=======

#TODO pass bound and apply
>>>>>>> Stashed changes
function multipropT1!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, all_params::AbstractMatrix{<:Real}, all_logdensity_values::Vector{<:Real}, proposed_params::AbstractMatrix, P_T1::AbstractVector{<:AbstractFloat}) # TODO include checks for input, optimize and write test
    indices(all_params, 2) != indices(all_logdensity_values, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(all_params, 2) != indices(P_T1, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of P_T1"))

    current_params = view(all_params, :, 1)  # memory allocation
    proposed_params = view(all_params, :, 2:end)  # memory allocation
    current_logdensity_value = all_logdensity_values[1]
    proposed_logdensity_values = view(all_logdensity_values, 2:end)  # memory allocation

    # Propose new parameters:
    proposal_rand!(rng, pdist, proposed_params, current_params)
    apply_bounds!(proposed_params, param_bounds(target), false)

    n_proposals = size(proposed_params, 2)
    n_proposals_inbounds = 0
    proposal_idxs_inbounds_tmp = Vector{Int}(n)  # Memory allocation
    fill!(proposal_idxs_inbounds_tmp, 0)
    @inbounds for j in indices(proposed_params, 2)
        if proposed_params[:, j] in param_bounds(target)
            n_proposals_inbounds += 1
            proposal_idxs_inbounds_tmp[n_proposals_inbounds] = j
        end
    end
    proposal_idxs_inbounds = proposal_idxs_inbounds_tmp[1:n_proposals_inbounds]  # Memory allocation
    proposed_params_inbounds = proposed_params[:, proposal_idxs_inbounds]  # Memory allocation

    # Evaluate target density at new parameters:
    proposed_logdensity_values_inbounds = similar(proposed_logdensity_values, n_proposals_inbounds)  # Memory allocation
    density_logval!(proposed_logdensity_values_inbounds, target, proposed_params_inbounds)
    fill!(proposed_logdensity_values, -Inf)
    proposed_logdensity_values[proposal_idxs_inbounds] = proposed_logdensity_values_inbounds

    all_params_inbounds = similar(proposed_params_inbounds, size(proposed_params_inbounds, 1), 1 + size(proposed_params_inbounds, 2))  # Memory allocation
    all_params_inbounds[:, 1] = current_params
    all_params_inbounds[:, 2:end] = proposed_params_inbounds

    # ToDo: Optimize for symmetric proposals?
    p_d_inbounds = similar(all_logdensity_values, size(all_params_inbounds, 2), size(all_params_inbounds, 2))
    distribution_logpdf!(view(p_d, :, j), pdist, all_params_inbounds, view(all_params_inbounds, :, j])  # Memory allocation due to view

    P_T1_inbounds = similar(P_T1, 1 + n_proposals_inbounds)  # Memory allocation

<<<<<<< Updated upstream
=======
=======
>>>>>>> Stashed changes
function multipropT1!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, params_old::AbstractVector, params_new::AbstractMatrix, P_T1::AbstractVector{<:AbstractFloat}) # TODO include checks for input, optimize and write test
    size(P_T1, 2) != 1 && throw(ArgumentError("The transition probability vector has wrong dimensions"))
    len = length(P_T1)
    size(params_new, 1) != size(params_old, 1) && throw(ArgumentError("The dimensions of the proposals are not correct"))
    size(params_new, 2) != len && throw(ArgumentError("The number of proposals is not correct"))

    #params_new = zeros(length(params_old), length(P_T1) - 1)
    p_d = zeros(len)
    p_t = zeros(len)
    proposal_rand!(rng, pdist, params_new, params_old)
    params_new[:, 1] .= params_old
    #params = cat(2, params_old, params_new)
    distribution_logpdf!(p_d, pdist, params_new, params_old)
    density_logval!(p_t, target, params_new)
<<<<<<< Updated upstream
>>>>>>> Stashed changes
=======
>>>>>>> Stashed changes
=======

>>>>>>> Stashed changes
    sum_log_d = sum_first_dim(p_d,1)
    P_T1_inbounds .= p_t - p_d + sum_log_d
    P_T1_inbounds .*=  inv(sum_first_dim(P_T1_inbounds,1))

    .....
<<<<<<< Updated upstream

    # P_T1, proposed_params  ....
end

<<<<<<< Updated upstream
<<<<<<< Updated upstream
multipropT1(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, current_params::AbstractVector, num_prop::Integer) = multipropT1!(rng, pdist, target, current_params, zeros(eltype(current_params), size(current_params, 1), num_prop + 1), zeros(num_prop + 1))
=======
multipropT1(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, params_old::AbstractVector, num_prop::Integer) = multipropT1!(rng, pdist, target, params_old, zeros(eltype(params_old), size(params_old, 1), num_prop + 1), zeros(num_prop + 1))
>>>>>>> Stashed changes
=======
multipropT1(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, params_old::AbstractVector, num_prop::Integer) = multipropT1!(rng, pdist, target, params_old, zeros(eltype(params_old), size(params_old, 1), num_prop + 1), zeros(num_prop + 1))
>>>>>>> Stashed changes


=======
end

multipropT1(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, current_params::AbstractVector, num_prop::Integer) = multipropT1!(rng, pdist, target, current_params, zeros(eltype(current_params), size(current_params, 1), num_prop + 1), zeros(num_prop + 1))

#TODO use rng in multiprop_transition, cumsum to a temp array,
>>>>>>> Stashed changes
function multiprop_transition!(P_T::AbstractVector{<:AbstractFloat}, proposed_params::AbstractMatrix, position::AbstractVector)
    size(position, 1) != size(proposed_params, 1) && throw(ArgumentError("The dimension of the new position vector is inconsistent with the data provided"))
    length(P_T) != size(proposed_params, 2) && throw(ArgumentError("The number of points provided is inconsitent with the number of proposals"))

    cumsum!(P_T, P_T, 1)
    prob = Distributions.Uniform()

    pos_ind = findfirst(x -> x >= prob, P_T1)

    position .= params[:, pos_ind]

    position

end

multiprop_transition(P_T::AbstractVector{<:AbstractFloat}, proposed_params::AbstractMatrix) = multiprop_transition!(P_T::AbstractVector{<:AbstractFloat}, proposed_params::AbstractMatrix, zeros(size(params, 1)))
