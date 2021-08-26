using AxisKeys
using Distributions
using InteractiveUtils
using LoopVectorization
using NamedDims
using Statistics
using Printf
using TensorOperations
using Tullio

export loo, psis_loo


#####################
###### STRUCTS ######
#####################


"""
    PsisLooMethod

Use Pareto-smoothed importance sampling together with leave-one-out cross validation to
estimate the out-of-sample predictive accuracy.
"""
struct PsisLooMethod <: AbstractCVMethod end


"""
    PsisLoo{
        F <: Real,
        AF <: AbstractArray{F},
        VF <: AbstractVector{F},
        I <: Integer,
        VI <: AbstractVector{I},
    } <: AbstractCV

A struct containing the results of jackknife (leave-one-out) cross validation using Pareto 
smoothed importance sampling.

$CV_DESC

See also: [`loo`]@ref, [`bayes_cv`]@ref, [`psis_loo`]@ref, [`Psis`]@ref
"""
struct PsisLoo{
    F <: Real,
    AF <: AbstractArray{F},
    VF <: AbstractVector{F},
    I <: Integer,
    VI <: AbstractVector{I},
} <: AbstractCV
    estimates::KeyedArray
    pointwise::KeyedArray
    psis_object::Psis{F, AF, VF, I, VI}
    mcse::F
end


function Base.show(io::IO, ::MIME"text/plain", loo_object::PsisLoo)
    table = loo_object.estimates
    _throw_pareto_k_warning(loo_object.pointwise(:pareto_k))
    post_samples = loo_object.psis_object.posterior_sample_size
    data_size = loo_object.psis_object.data_size
    println(
        "Results of PSIS-LOO-CV with $post_samples Monte Carlo samples and $data_size " *
        Printf.@sprintf("data points. Total Monte Carlo SE of %.2g.", loo_object.mcse),
    )
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.criterion,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end


#####################
#### LOO METHODS ####
#####################


"""
    function loo(args...; method=PsisLooMethod(), kwargs...) -> PsisLoo

Compute the approximate leave-one-out cross-validation score using the specified method.

Currently, this function only serves to call `psis_loo`, but this could change in the
future. The default methods or return type may change without warning; thus, we recommend
using `psis_loo` instead if reproducibility is required.

See also: [`psis_loo`](@ref), [`PsisLoo`](@ref).
"""
function loo(args...; method=PsisLooMethod(), kwargs...)
    return psis_loo(args...; kwargs...)
end


"""
    function psis_loo(
        log_likelihood::Array{Real} [, args...];
        [, chain_index::Vector{Integer}, kwargs...]
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score.

# Arguments

  - `log_likelihood::Array`: A matrix or 3d array of log-likelihood values indexed as
    `[data, step, chain]`. The chain argument can be left off if `chain_index` is provided
    or if all posterior samples were drawn from a single chain.
  - $ARGS [`psis`](@ref).
  - $CHAIN_INDEX_DOC
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(
    log_likelihood::AbstractArray{<:Real, 3}, args...; kwargs...
)

    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    # log_likelihood::ArrayType = similar(log_likelihood)
    # log_likelihood .= score(log_likelihood)

    psis_object = psis(-log_likelihood, args...; kwargs...)
    weights = psis_object.weights
    ξ = psis_object.pareto_k
    r_eff = psis_object.r_eff

    @tullio pointwise_loo[i] := weights[i, j, k] * exp(log_likelihood[i, j, k]) |> log
    @tullio pointwise_naive[i] := exp(log_likelihood[i, j, k] - log_count) |> log
    pointwise_p_eff = pointwise_naive - pointwise_loo
    pointwise_mcse = _calc_mcse(weights, log_likelihood, pointwise_loo, r_eff)


    pointwise = KeyedArray(
        hcat(pointwise_loo, pointwise_naive, pointwise_p_eff, pointwise_mcse, ξ);
        data=1:length(pointwise_loo),
        statistic=[:cv_est, :naive_est, :p_eff, :mcse, :pareto_k],
    )

    table = _generate_loo_table(pointwise)

    @tullio mcse := pointwise_mcse[i]^2
    mcse = sqrt(mcse)

    return PsisLoo(table, pointwise, psis_object, mcse)

end


function psis_loo(
    log_likelihood::AbstractMatrix{<:Real},
    args...;
    chain_index::AbstractVector=ones(size(log_likelihood, 1)),
    kwargs...,
)
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return psis_loo(new_log_ratios, args...; kwargs...)
end


function _generate_loo_table(pointwise::AbstractArray{<:Real})

    data_size = size(pointwise, :data)
    # create table with the right labels
    table = KeyedArray(
        similar(NamedDims.unname(pointwise), 3, 4);
        criterion=[:cv_est, :naive_est, :p_eff],
        statistic=[:total, :se_total, :mean, :se_mean],
    )

    # calculate the sample expectation for the total score
    to_sum = pointwise([:cv_est, :naive_est, :p_eff])
    @tullio averages[crit] := to_sum[data, crit] / data_size
    averages = reshape(averages, 3)
    table(:, :mean) .= averages

    # calculate the sample expectation for the average score
    table(:, :total) .= table(:, :mean) .* data_size

    # calculate the sample expectation for the standard error in the totals
    se_mean = std(to_sum; mean=averages', dims=1) / sqrt(data_size)
    se_mean = reshape(se_mean, 3)
    table(:, :se_mean) .= se_mean

    # calculate the sample expectation for the standard error in averages
    table(:, :se_total) .= se_mean * data_size

    if table(:p_eff, :total) ≤ 0
        @warn "The calculated effective number of parameters is negative, which should " *
        "not be possible. PSIS has failed to approximate the target distribution."
    end

    return table
end


function _calc_mcse(weights, log_likelihood, pointwise_loo, r_eff)
    E_epd = exp.(pointwise_loo)
    @tullio pointwise_var[i] := 
        (weights[i, j, k] * (exp(log_likelihood[i, j, k]) - E_epd[i]))^2
    # apply autocorrelation adjustment:
    # If MCMC draws follow a log-normal distribution, then their log has this std. error:
    @turbo @. pointwise_var = log1p(pointwise_var / E_epd^2)
    # (google "log-normal method of moments" for a proof)
    # apply MCMC correlation correction:
    return @turbo @. sqrt(pointwise_var / r_eff)
end
