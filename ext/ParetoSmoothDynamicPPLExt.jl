module ParetoSmoothDynamicPPLExt

if isdefined(Base, :get_extension)
    using ParetoSmooth
    using MCMCChains
    using DynamicPPL
else
    using ..ParetoSmooth
    using ..MCMCChains
    using ..DynamicPPL
end

const TURING_MODEL_ARG = """
`model`: A Turing model with data in the form of `model(data)`.
"""

using ParetoSmooth: LIKELIHOOD_FUNCTION_ARG, DATA_ARG, CHAINS_ARG, ARGS, KWARGS

"""
    pointwise_log_likelihoods(model::DynamicPPL.Model, chains::Chains) -> Array

Compute pointwise log-likelihoods from a Turing model.

# Arguments
  - $TURING_MODEL_ARG
  - $CHAINS_ARG

# Returns
  - `Array`: A three dimensional array of pointwise log likelihoods. This array should be
    indexed using `array[data, sample, chains]`.
"""
function ParetoSmooth.pointwise_log_likelihoods(model::DynamicPPL.Model, chains::Chains)
    # subset of chains for mcmc samples
    chain_params = MCMCChains.get_sections(chains, :parameters)
    # compute the pointwise log likelihoods
    log_like_dict = DynamicPPL.pointwise_loglikelihoods(model, chain_params)
    # Size of array (n_steps, n_chains) using first parameter
    dims = size(last(first(log_like_dict)))
    # parse "var[i]" -> i
    ind_from_string(x) = parse(Int, rsplit(rsplit(x, "[", limit=2)[2], "]", limit=2)[1])
    # collect variable names
    sorted_keys = sort(collect(keys(log_like_dict)); by=ind_from_string)
    # Convert from dictionary to 3d array
    array = [reshape(log_like_dict[i], 1, dims...) for i in sorted_keys]
    return reduce(vcat, array)
end


"""
    psis_loo(model::DynamicPPL.Model, chains::Chains, args...; kwargs...) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score from a `chains` object and a Turing model.

# Arguments

  - $CHAINS_ARG
  - $TURING_MODEL_ARG
  - $ARGS [`psis`](@ref).
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function ParetoSmooth.psis_loo(model::DynamicPPL.Model, chains::Chains, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(model, chains)
    return psis_loo(pointwise_log_likes, args...; kwargs...)
end


"""
    loo_from_psis(model::DynamicPPL.Model, chains::Chains, args...; kwargs...) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score from a `Chains` object, a Turing model, and a precalculated `Psis` object.

# Arguments

  - $CHAINS_ARG
  - $TURING_MODEL_ARG
  - `psis`: A `Psis` object containing the results of Pareto smoothed importance sampling.

See also: [`psis`](@ref), [`psis_loo`](@ref), [`PsisLoo`](@ref).
"""
function ParetoSmooth.loo_from_psis(model::DynamicPPL.Model, chains::Chains, psis::Psis)
    pointwise_log_likes = pointwise_log_likelihoods(model, chains)
    return loo_from_psis(pointwise_log_likes, psis)
end


"""
    psis(model::DynamicPPL.Model, chains::Chains, args...; kwargs...) -> Psis

Generate samples using Pareto smoothed importance sampling (PSIS).

# Arguments
  - $TURING_MODEL_ARG
  - $CHAINS_ARG

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function ParetoSmooth.psis(model::DynamicPPL.Model, chains::Chains, args...; kwargs...)
    log_ratios = pointwise_log_likelihoods(model, chains)
    return psis(-log_ratios, args...; kwargs...)
end


function ParetoSmooth.naive_lpd(model::DynamicPPL.Model, chains::Chains, args...; kwargs...)
    log_ratios = pointwise_log_likelihoods(model, chains)
    return ParetoSmooth.naive_lpd(log_ratios, args...; kwargs...)
end

end
