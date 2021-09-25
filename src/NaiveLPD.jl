using StatsFuns

"""
    naive_lpd(log_likelihood::AbstractArray{<:Real})

Calculate the naive (in-sample) estimate of the expected log probability density, otherwise
known as the in-sample Bayes score. Not recommended for most uses.

# Arguments
  - $LOG_LIK_ARR
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
    mcmc_count = dims[2] * dims[3]
    log_count = log(mcmc_count)

    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    return @tullio naive := pointwise_naive[i]
end
