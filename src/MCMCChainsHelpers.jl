using .MCMCChains
export pointwise_log_likelihoods

"""
    pointwise_log_likelihoods(ll_fun::Function, chains::Chains, data)

Compute the pointwise log likelihoods.

# Arguments
  - $LIKELIHOOD_FUNCTION_ARG
  - `chain::Chains`: A chain object from MCMCChains.
  - `data`: An array of data points used to estimate the parameters of the model.

# Returns
  - `Array`: a three dimensional array of pointwise log-likelihoods. Dimensions are ordered
    as `[data, step, chain]`.
"""
function pointwise_log_likelihoods(ll_fun::Function, chain::Chains, data::AbstractVector; kwargs...)
    samples = Array(Chains(chain, :parameters).value)
    pointwise_log_likelihoods(ll_fun, samples, data; kwargs...)
end

"""
    function psis_loo(
        ll_fun::Function, 
        chain::Chains, 
        data, 
        args...; 
        kwargs...
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score from an MCMCChains object.

# Arguments

  - $LIKELIHOOD_FUNCTION_ARG
  - `chain::Chain`: A chain object from MCMCChains.
  - `data`: A vector of data points used to estimate the parameters of the model.
  - $ARGS [`psis_loo`](@ref).
  - $KWARGS [`psis_loo`](@ref).

See also: [`psis_loo`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(ll_fun::Function, chain::Chains, data::AbstractVector, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis_loo(pointwise_log_likes, args...; kwargs...)
end

"""
    psis(ll_fun::Function, chain::Chains, data[, kwargs...; args...) -> Psis

Implements Pareto-smoothed importance sampling (PSIS) based on MCMCChain object.

# Arguments

  - $LIKELIHOOD_FUNCTION_ARG
  - `chain::Chain`: A chain object from MCMCChains.
  - `data`: A vector of data points used to estimate the parameters of the model.
  - $ARGS [`psis`](@ref).
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`psis_loo`](@ref), [`PsisLoo`](@ref).
"""
function psis(ll_fun::Function, chain::Chains, data::AbstractVector, args...; kwargs...)
    pointwise_log_likes = pointwise_log_likelihoods(ll_fun, chain, data)
    return psis(-pointwise_log_likes, args...; kwargs...)
end