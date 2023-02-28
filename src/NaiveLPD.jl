"""
    naive_lpd(log_likelihood::AbstractArray{<:Real}[, chain_index])

Calculate the naive (in-sample) estimate of the expected log probability density, otherwise
known as the in-sample Bayes score. This method yields heavily biased results, and we advise
against using it; it is included only for pedagogical purposes.

This method is unexported and can only be accessed by calling `ParetoSmooth.naive_lpd`.

# Arguments
  - $LOG_LIK_ARR
  - $CHAIN_INDEX_DOC
"""
function naive_lpd(log_likelihood::AbstractArray{<:Real, 3})
    @info "We advise against using `naive_lpd`, as it gives inconsistent and strongly " *
    "biased estimates. Use `psis_loo` instead."
    return _naive_lpd(log_likelihood)
end


function naive_lpd(
    log_likelihood::AbstractMatrix{<:Real}, 
    chain_index::AbstractVector = _assume_one_chain(log_likelihood)
)
    chain_index = Int.(chain_index)
    log_likelihood = _convert_to_array(log_likelihood, chain_index)
    return naive_lpd(log_likelihood)
end


function _naive_lpd(log_likelihood::AbstractArray{<:Real, 3}) 
    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]
    log_count = log(mcmc_count)

    pointwise_naive = zeros(eltype(log_likelihood), data_size)
    for k = axes(log_likelihood, 3), j = axes(log_likelihood, 2), i = axes(log_likelihood, 1)
        @inbounds pointwise_naive[i] += exp_inline(log_likelihood[i, j, k] - log_count)
    end
    naive = zero(eltype(pointwise_naive))
    for i = eachindex(pointwise_naive)
        naive += log(pointwise_naive[i])
    end
    return naive
end
