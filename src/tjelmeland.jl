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

function multi_propose!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, all_params::AbstractMatrix{<:Real}, is_inbounds::BitVector) # TODO include checks for input, optimize and write test
    indices(all_params, 2) != indices(is_inbounds, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of inbound bools"))
    current_params = view(all_params, :, 1)  # memory allocation
    proposed_params = view(all_params, :, 2:size(all_params, 2))  # memory allocation

    # Propose new parameters:
    proposal_rand!(rng, pdist, proposed_params, current_params)
    apply_bounds!(proposed_params, param_bounds(target), false)

    is_inbounds .= false
    is_inbounds[1] = true
    n_proposals = size(proposed_params, 2)
    n_proposals_inbounds = 0
    proposal_idxs_inbounds_tmp = Vector{Int}(n_proposals)  # Memory allocation
    fill!(proposal_idxs_inbounds_tmp, 0)
    @inbounds for j in indices(proposed_params, 2)
        if proposed_params[:, j] in param_bounds(target)
            is_inbounds[j+1] = true
        end
    end

end

function multipropT1!(rng::AbstractRNG, pdist::GenericProposalDist, target::AbstractDensity, all_params::AbstractMatrix{<:Real}, all_logdensity_values::Vector{<:Real}, is_inbounds::BitVector, P_T1::AbstractVector{<:AbstractFloat}) # TODO include checks for input, optimize and write test
    indices(all_params, 2) != indices(all_logdensity_values, 1) && throw(ArgumentError("Number of parameter sets doesn't match number of log(density) values"))
    indices(all_params, 2) != indices(P_T1, 1) && throw(ArgumentError("Number of parameter sets doesn't match size of P_T1"))

    # Evaluate target density at new parameters:
    proposed_logdensity_values = view(all_logdensity_values, 2:size(all_logdensity_values, 1))
    fill!(proposed_logdensity_values, -Inf)
    proposed_logdensity_values_inbounds = view(proposed_logdensity_values, is_inbounds[2:end])  # Memory allocation
    density_logval!(proposed_logdensity_values_inbounds, target, proposed_params_inbounds)
    all_logdensity_values_inbounds = view(all_logdensity_values, is_inbounds)

    all_params_inbounds = view(all_params, :, is_inbounds)

    # ToDo: Optimize for symmetric proposals?
    p_d_inbounds = similar(all_logdensity_values, size(all_params_inbounds, 2), size(all_params_inbounds, 2))
    @inbounds for j in indices(all_params_inbounds, 2)
        distribution_logpdf!(view(p_d, :, j), pdist, all_params_inbounds, view(all_params_inbounds, :, j))  # Memory allocation due to view
    end

    P_T1_inbounds = view(P_T1, is_inbounds)  # Memory allocation

    @inbounds for j in indices(P_T1_inbounds)
        P_T1_inbounds[j] = exp(sum_first_dim(p_d_inbounds, j) - p_d_inbounds[j] + all_logdensity_values_inbounds[j])
    end

    P_T1_inbounds .*=  inv(sum_first_dim(P_T1_inbounds,1))
    P_T1[!is_inbounds] .= 0.0

end


function multiprop_select!(rng::AbstractRNG, P_T::AbstractVector{<:AbstractFloat}, all_params::AbstractMatrix{<:Real}, is_inbounds::AbstractVector{<:Bool})
    indices(all_params_old, 2) != indices(all_logdensity_values, 1) && throw(ArgumentError("The dimension of the new position vector is inconsistent with the data provided"))
    indices(all_params_old, 2) != indices(P_T, 1) && throw(ArgumentError("The number of points provided is inconsistent with the number of proposals"))
    indices(all_params_old, 1) != indices(all_params_new, 1) && throw(ArgumentError("Old and New have inconsistent number of rows"))
    indices(all_params_old, 2) != indices(all_params_new, 2) && throw(ArgumentError("Old and New have inconsistent number of columns"))
    !(sum_first_dim(P_T) ≈ 1.) && throw(ArgumentError("The transition probabilities do not sum up to 1."))

    cms = cumsum(P_T, 1)  # Memory allocation!
    prob = rand(rng)

    pos_ind = findfirst(x -> x >= prob, cms)
    @assert is_inbounds[pos_ind]

    #TODO temp array or new memory alloc??
    #all_params_new[:, 1] = all_params_old[:, pos_ind]
    #all_params_new[:, pos_ind] = all_params_old[:, 1]
    #rest = setdiff(indices(all_params_old, 2),(1,pos_ind))
    #all_params_new[:, rest] = all_params_old[:, rest]
    # TODO implement weighting scheme ... should I swap weights?? ... should I swap all_logdensity_values ??

    all_params_new
    return pos_ind
end
