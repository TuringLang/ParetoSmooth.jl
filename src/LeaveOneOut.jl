using AxisKeys
using InteractiveUtils
using NamedDims
using Statistics
using Printf
using Tullio

export loo, psis_loo, loo_from_psis, PsisLoo


#####################
###### STRUCTS ######
#####################


# """
#     PsisLooMethod

# Use Pareto-smoothed importance sampling together with leave-one-out cross validation to
# estimate the out-of-sample predictive accuracy.
# """
# struct PsisLooMethod <: AbstractCVMethod end


"""
    PsisLoo <: AbstractCV

A struct containing the results of leave-one-out cross validation computed with Pareto 
smoothed importance sampling.

$CV_DESC

See also: [`loo`]@ref, [`bayes_cv`]@ref, [`psis_loo`]@ref, [`Psis`]@ref
"""
struct PsisLoo{
    RealType<:Real,
    ArrayType<:AbstractArray{RealType},
    VectorType<:AbstractVector{RealType},
} <: AbstractCV
    estimates::KeyedArray{RealType,2,<:NamedDimsArray,<:Any}
    pointwise::KeyedArray{RealType,2,<:NamedDimsArray,<:Any}
    psis_object::Psis{RealType,ArrayType,VectorType}
    gmpd::RealType
    mcse::RealType
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
        table.data;
        compact_printing=false,
        header=table.column,
        row_names=table.statistic,
        formatters=ft_printf("%5.2f"),
        alignment=:r
    )
end


#####################
#### LOO METHODS ####
#####################


"""
    function loo(args...; kwargs...) -> PsisLoo

Compute an approximate leave-one-out cross-validation score.

Currently, this function only serves to call `psis_loo`, but this could change in the
future. The default methods or return type may change without warning, so we recommend
using `psis_loo` instead if reproducibility is required.

See also: [`psis_loo`](@ref), [`PsisLoo`](@ref).
"""
function loo(args...; kwargs...)
    return psis_loo(args...; kwargs...)
end


"""
    function psis_loo(
        log_likelihood::AbstractArray{<:Real} [, args...];
        [, chain_index::Vector{Int}, kwargs...]
    ) -> PsisLoo

Use Pareto-Smoothed Importance Sampling to calculate the leave-one-out cross validation
score.

# Arguments

  - $LOG_LIK_ARR
  - $ARGS [`psis`](@ref).
  - $CHAIN_INDEX_DOC
  - $KWARGS [`psis`](@ref).

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).
"""
function psis_loo(log_likelihood::AbstractArray{<:Real,3}, args...; kwargs...)
    psis_object = psis(-log_likelihood, args...; kwargs...)
    return loo_from_psis(log_likelihood, psis_object)
end


function psis_loo(
    log_likelihood::AbstractMatrix{<:Real},
    args...;
    chain_index::AbstractVector=_assume_one_chain(log_likelihood),
    kwargs...
)
    chain_index = Int.(chain_index)
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return psis_loo(new_log_ratios, args...; kwargs...)
end


"""
    loo_from_psis(
        log_likelihood::AbstractArray{<:Real}, psis_object::Psis; 
        chain_index::Vector{<:Integer}
    )

Use a precalculated `Psis` object to estimate the leave-one-out cross validation score.

# Arguments

  - $LOG_LIK_ARR
  - `psis_object`: A precomputed `Psis` object used to estimate the LOO-CV score.
  - $CHAIN_INDEX_DOC

See also: [`psis`](@ref), [`loo`](@ref), [`PsisLoo`](@ref).

"""
function loo_from_psis(log_likelihood::AbstractArray{<:Real,3}, psis_object::Psis)
    dims = size(log_likelihood)
    data_size = dims[1]
    mcmc_count = dims[2] * dims[3]  # total number of samples from posterior
    log_count = log(mcmc_count)


    # TODO: Add a way of using score functions other than ELPD
    # log_likelihood::ArrayType = similar(log_likelihood)
    # log_likelihood .= score(log_likelihood)


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
        statistic=[:cv_elpd, :naive_lpd, :p_eff, :mcse, :pareto_k]
    )

    table = _generate_loo_table(pointwise)

    gmpd = exp.(table(column=:mean, statistic=:cv_elpd))

    mcse = sum(abs2, pointwise_mcse) |> sqrt
    return PsisLoo(table, pointwise, psis_object, gmpd, mcse)
end


function loo_from_psis(
    log_likelihood::AbstractMatrix{<:Real}, psis_object::Psis, args...;
    chain_index::AbstractVector=_assume_one_chain(log_likelihood), kwargs...
)
    chain_index = Int.(chain_index)
    new_log_ratios = _convert_to_array(log_likelihood, chain_index)
    return loo_from_psis(new_log_ratios, psis_object, args...; kwargs...)
end


function _generate_loo_table(pointwise::AbstractMatrix{<:Real})

    data_size = size(pointwise, :data)
    # create table with the right labels
    table = KeyedArray(
        similar(NamedDims.unname(pointwise), 3, 4);
        statistic=[:cv_elpd, :naive_lpd, :p_eff],
        column=[:total, :se_total, :mean, :se_mean]
    )

    # calculate the sample expectation for the total score
    to_sum = pointwise([:cv_elpd, :naive_lpd, :p_eff])
    @tullio avgs[statistic] := to_sum[data, statistic] |> _ / data_size
    avgs = reshape(avgs, 3)
    table(:, :mean) .= avgs

    # calculate the sample expectation for the average score
    table(:, :total) .= table(:, :mean) .* data_size

    # calculate the sample expectation for the standard error in the totals
    se_mean = std(to_sum; mean=avgs', dims=1) / sqrt(data_size)
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
    pointwise_gmpd = exp.(pointwise_loo)
    @tullio pointwise_var[i] :=
        (weights[i, j, k] * (exp(log_likelihood[i, j, k]) - pointwise_gmpd[i]))^2
    # If MCMC draws follow a log-normal distribution, then their log has this std. error:
    @. pointwise_var = log1p(pointwise_var / pointwise_gmpd^2)
    # (google "log-normal method of moments" for a proof)
    # apply MCMC correlation correction:
    return @. sqrt(pointwise_var / r_eff)
end
