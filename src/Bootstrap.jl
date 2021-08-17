using AxisKeys
using InteractiveUtils
using LoopVectorization
using Random
using Statistics
using Tullio

export bayes_cv

"""
    function bayes_cv(
        log_likelihood::Array{Float} [, args...];
        source::String="mcmc" [, chain_index::Vector{Int}, kwargs...]
    ) -> PsisBB

Use the Bayesian bootstrap (Bayes cross-validation) and PSIS to calculate an approximate
posterior distribution for the out-of-sample score.


# Arguments

  - `log_likelihood::Array`: An array or matrix of log-likelihood values indexed as
    `[data, step, chain]`. The chain argument can be left off if `chain_index` is provided
    or if all posterior samples were drawn from a single chain.
  - `args...`: Positional arguments to be passed to [`psis`](@ref).
  - `chain_index::Vector`: An (optional) vector of integers specifying which chain each
    step belongs to. For instance, `chain_index[3]` should return `2` if
    `log_likelihood[:, 3]` belongs to the second chain.
  - `kwargs...`: Keyword arguments to be passed to [`psis`](@ref).


# Extended help
The Bayesian bootstrap works similarly to other cross-validation methods: First, we remove
some piece of information from the model. Then, we test how well the model can reproduce 
that information. With leave-k-out cross validation, the information we leave out is the
value for one or more data points. With the Bayesian bootstrap, the information being left
out is the true probability of each observation.


See also: [`BayesCV`](@ref), [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function bayes_cv(
    log_likelihood::T, 
    args...;
    resamples::Integer=2^10, 
    rng=MersenneTwister(1865),
    kwargs...
) where {F<:AbstractFloat, T<:AbstractArray{F, 3}}

    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    bb_weights = data_size * rand(rng, Dirichlet(ones(data_size)), resamples)
    @tullio bb_samples[re, step, chain] := 
        bb_weights[datum, re] * log_likelihood[datum, step, chain]
    @tullio log_is_ratios[re, step, chain] := 
        (bb_weights[datum, re] - 1) * log_likelihood[datum, step, chain]
    psis_object = psis(log_is_ratios, args...; kwargs...)
    psis_weights = psis_object.weights

    @tullio re_naive[re] := log <| # calculate the naive estimate in many resamples
        psis_weights[re, step, chain] * exp(bb_samples[re, step, chain])
    @tullio sample_est[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    @tullio naive_est := sample_est[i]

    bb_ests = (2 * naive_est) .- re_naive
    @tullio mcse[re] := sqrt <|
        (psis_weights[re, step, chain] * (bb_samples[re, step, chain] - re_naive[re]))^2
    bootstrap_se = std(re_naive) / sqrt(resamples)

    # Posterior for the *average score*, not the mean of the posterior distribution:
    resample_calcs = KeyedArray(
        hcat(
            bb_ests,
            re_naive,
            re_naive - bb_ests,
            mcse,
            psis_object.pareto_k
        );
        data=Base.OneTo(resamples),
        statistic=[
            :loo_est,
            :naive_est,
            :overfit,
            :mcse,
            :pareto_k
        ],
    )

    estimates = _generate_bayes_table(log_likelihood, resample_calcs, resamples, data_size)

    return BayesCV(
        estimates,
        resample_calcs,
        psis_object,
        data_size
    )

end


function bayes_cv(
    log_likelihood::T,
    args...;
    chain_index::AbstractVector=ones(size(log_likelihood, 1)),
    kwargs...,
) where {F<:AbstractFloat, T<:AbstractMatrix{F}}
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return psis_loo(new_log_ratios, args...; kwargs...)
end


function _generate_bayes_table(
    log_likelihood::AbstractArray, 
    pointwise::AbstractArray, 
    resamples::Integer,
    data_size::Integer
)

    # create table with the right labels
    table = KeyedArray(
        similar(log_likelihood, 3, 4);
        criterion=[:loo_est, :naive_est, :overfit],
        statistic=[:total, :se_total, :mean, :se_mean],
    )

    # calculate the sample expectation for the total score
    to_sum = pointwise([:loo_est, :naive_est, :overfit])
    @tullio totals[crit] := to_sum[re, crit] / resamples
    totals = reshape(totals, 3)
    table(:, :total) .= totals

    # calculate the sample expectation for the average score
    table(:, :mean) .= table(:, :mean) / data_size

    # calculate the sample expectation for the standard error in the totals
    se_total = std(to_sum; dims=1) * sqrt(data_size)
    se_total = reshape(se_total, 3)
    table(:, :se_total) .= se_total

    # calculate the sample expectation for the standard error in averages
    table(:, :se_mean) .= se_total / data_size

    return table
end
