using AxisKeys
using MeasureTheory
using InteractiveUtils
using LoopVectorization
using Statistics
using Tullio

export bayes_val

"""
    function bayes_cv(
        log_likelihood::Array{Float} [, args...];
        source::String="mcmc" [, chain_index::Vector{Int}, kwargs...]
    ) -> PsisBB

Use the Bayesian bootstrap (Bayes cross-validation) and PSIS to calculate an approximate
posterior for the out-of-sample score.


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
    rng=MersenneTwister(1776),
    kwargs...
) where {F<:AbstractFloat, T<:AbstractArray{F, 3}}

    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    bb_weights = data_size * rand(rng, Dirichlet(ones(data_size)), resamples)
    bb_samples = similar(log_likelihood, (resamples, data_size))
    psis_object = psis(bb_samples, args...; kwargs...)

    # "Pointwise" used here to refer to "per resample"
    @tullio pointwise_naive[i] := log <|
        psis_object.weights[i, j, k] * exp(bb_samples[i, j, k])
    @tullio sample_est := exp(log_likelihood[i, j, k] - log_count)
    sample_est = log(sample_est)
    
    @tturbo bb_ests .= (2 * sample_est) .- pointwise_naive
    @tullio pointwise_mcse[i] :=  # I'll take sqrt later in-place
        (weights[i, j, k] * (log_likelihood[i, j, k] - pointwise_loo[i]))^2
    # Apply law of total variance
    bootstrap_se = var(naive_ests) / bb_samples
    mcse = sqrt(mean(pointwise_mcse) + bootstrap_se)
    @tturbo @. pointwise_mcse = sqrt(pointwise_mcse)
        
    # Posterior for the *average score*, not the mean of the posterior distribution:
    posterior_avg = bb_ests / data_size
    resample_calcs = KeyedArray(
        hcat(
            bb_ests,
            pointwise_naive,
            pointwise_overfit,
            pointwise_mcse,
            psis_object.pareto_k
        );
        data=1:length(pointwise_loo),
        statistic=[
            :loo_est,
            :naive_est,
            :overfit,
            :mcse,
            :pareto_k
        ],
    )

    estimates = _generate_bayes_table(log_likelihood, resample_calcs, data_size)

    return BayesCV(
        estimates,
        resample_calcs,
        psis_object
    )

end


function bayes_cv(
    log_likelihood::T,
    args...;
    chain_index::AbstractVector=ones(size(log_likelihood, 1)),
    kwargs...,
) where {F <: AbstractFloat, T <: AbstractMatrix{F}}
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return psis_loo(new_log_ratios, args...; kwargs...)
end


function _generate_bayes_table(
    log_likelihood::AbstractArray, 
    pointwise::AbstractArray, 
    data_size::Integer
)

    # create table with the right labels
    table = KeyedArray(
        similar(log_likelihood, 3, 4);
        criterion=[:cv_est, :naive_est, :overfit],
        statistic=[:ev_total, :se_total, :ev_mean, :se_mean, :sd_mean],
    )
    
    # calculate the sample expectation for the total score
    to_sum = pointwise([:loo_est, :naive_est])
    @tullio total[crit] := to_sum[data, crit]
    table(:, :total) .= reshape(total, 3)

    # calculate the sample expectation for the average score
    table(:, :mean) .= table(:, :total) ./ data_size

    # calculate the sample expectation for the standard error in the totals
    @_ table(:, :se_total) .= pointwise([:loo_est, :naive_est, :overfit]) |> 
        varm(_, table(:, :mean); dims=1) |>
        sqrt.(data_size * _) |>
        reshape(_, 3)

    # calculate the sample expectation for the standard error in averages
    table(:, :se_mean) .= table(:, :se_total) ./ data_size

    return table
end
