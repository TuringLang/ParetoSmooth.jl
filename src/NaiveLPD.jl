using LoopVectorization
using Tullio


"""
    naive_lpd(log_likelihood::AbstractArray{<:Real, 3})

Calculate the naive (in-sample) estimate of the expected log probability density, otherwise
known as the in-sample Bayes score. Not recommended for most uses.
"""
function naive_lpd(log_likelihood::AbstractArray{<:Real, 3})

    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)

    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log

    return sum(pointwise_naive)
end
