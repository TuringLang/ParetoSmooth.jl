using StatsFuns
using LoopVectorization
import Base.show

export loo_compare, ModelComparison

const LOO_COMPARE_KWARGS = """
- `model_names`: A vector or tuple of strings or symbols used to identify models. If
none, models are numbered using the order of the arguments.
- `sort_models`: Sort models by total score.
- `high_to_low`: Sort models from best to worst score. If `false`, reverse the order.
"""

"""
    ModelComparison

A struct containing the results of model comparison.

# Fields

  - `pointwise::KeyedArray`: An array containing 
  - `estimates::KeyedArray`: A table containing the results of model comparison, with the
    following columns --
      + `cv_elpd`: The difference in total leave-one-out cross validation scores
        between models.
      + `cv_avg`: The difference in average LOO-CV scorees between models.
      + `weight`: A set of Akaike-like weights assigned to each model, which can be used in
        pseudo-Bayesian model averaging.
  - `std_err::NamedTuple`: A named tuple containing the standard error of `cv_elpd`. Note 
    that these estimators (incorrectly) assume that all folds are independent, despite their 
    substantial overlap, which creates a severely biased estimator. In addition, note 
    that LOO-CV differences are *not* asymptotically normal. As a result, these standard 
    errors cannot be used to calculate a confidence interval. These standard errors are 
    included for consistency with R's LOO package, and should not be relied upon.

See also: [`PsisLoo`](@ref)
"""
struct ModelComparison
    pointwise::KeyedArray
    estimates::KeyedArray
    std_err::NamedTuple
end



"""
    function loo_compare(
        cv_results::PsisLoo...;
        sort_models::Bool=true,
        best_to_worst::Bool=true,
        [, model_names::Tuple{Symbol}]
    ) -> ModelComparison

Construct a model comparison table from several [`PsisLoo`](@ref) objects.

# Arguments

  - `cv_results`: One or more [`PsisLoo`](@ref) objects to be compared. Alternatively,
  a tuple or named tuple of `PsisLoo` objects can be passed. If a named tuple is passed,
  these names will be used to label each model. 
  - $LOO_COMPARE_KWARGS

See also: [`ModelComparison`](@ref), [`PsisLoo`](@ref), [`psis_loo`](@ref)
"""
function loo_compare(
    cv_results::AbstractVector{<:PsisLoo};
    model_names::StringContainer=[Symbol("model_$i") for i in 1:length(cv_results)],
    sort_models::Bool=true,
    high_to_low::Bool=true,
) where {StringContainer <: Base.AbstractVecOrTuple{<:Union{AbstractString, Symbol}}}

    model_names = [Symbol(model_names[i]) for i in 1:length(model_names)]  # array version
    n_models, data_size = _get_dims(cv_results, model_names)
    
    if sort_models
        sorting_by = x -> x.estimates(:cv_elpd, :total)
        order = sortperm(cv_results; by=sorting_by, rev=high_to_low)
        permute!(cv_results, order)
        permute!(model_names, order)
    end

    # Extract relevant values from PsisLoo objects
    @views begin
        cv_elpd = [cv_results[i].estimates(:cv_elpd, :total) for i in 1:n_models]
        pointwise = [cv_results[i].pointwise for i in 1:n_models]
    end


    # Construct pointwise differences
    pointwise_diffs = cat(pointwise...; dims=3)
    pointwise_diffs = KeyedArray(
        pointwise_diffs;
        data=1:data_size,
        statistic=[:cv_elpd, :naive_lpd, :p_eff, :mcse, :pareto_k],
        model=model_names,
    )

    # Subtract the effective number of params and elpd ests; leave mcse+pareto_k the same
    @views base_case = pointwise_diffs[:, 1:3, 1]
    @turbo @. pointwise_diffs[:, 1:3, Not(1)] = pointwise_diffs[:, 1:3, Not(1)] - base_case
    pointwise_diffs[:, 1:3, 1] .= 0

    name_tuple = ntuple(i -> model_names[i], n_models)  # convert to tuple

    variance = var(pointwise_diffs(data=:, statistic=:cv_elpd, model=:); dims=:data)
    se_total = @. sqrt(data_size * variance)
    se_total = ntuple(i -> se_total[i], n_models)
    se_total = NamedTuple{name_tuple}(se_total)

    log_norm = logsumexp(cv_elpd)
    weights = @turbo @. exp(cv_elpd - log_norm)
    avg_elpd = @turbo @. cv_elpd / data_size
    @turbo @. cv_elpd = cv_elpd - cv_elpd[1]
    total_diffs = KeyedArray(
        hcat(cv_elpd, avg_elpd, weights);
        model=model_names,
        statistic=[:cv_elpd, :cv_avg, :weight],
    )
    
    return ModelComparison(pointwise_diffs, total_diffs, se_total)

end


function loo_compare(cv_results::NamedTuple; kwargs...)
    @nospecialize cv_results
    return loo_compare(values(cv_results); model_names=keys(cv_results), kwargs...)
end


function loo_compare(cv_results::Tuple; kwargs...)
    @nospecialize cv_results
    @views cv_results = [cv_results[i] for i in 1:length(cv_results)]
    return loo_compare(cv_results; kwargs...)
end


function loo_compare(cv_results::PsisLoo...; kwargs...)
    @nospecialize cv_results
    @views cv_results = [cv_results[i] for i in 1:length(cv_results)]
    return loo_compare(cv_results; kwargs...)
end


function Base.show(io::IO, ::MIME"text/plain", model_comparison::ModelComparison)
    estimates = model_comparison.estimates
    return pretty_table(
        estimates;
        compact_printing=false,
        header=estimates.statistic,
        row_names=estimates.model,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end


function _get_dims(cv_results::AbstractVector{<:PsisLoo}, model_names::Vector{Symbol})

    n_models = length(cv_results)
    if n_models != length(model_names)
        throw(ArgumentError("Number of model names does not match size of `cv_results`."))
    end

    data_sizes = [cv_results[i].psis_object.data_size for i in 1:n_models]
    data_size = data_sizes[1]
    if any(data_sizes .≠ data_size)
        throw(
            ArgumentError(
                "Number of data points differs across `PsisLoo` objects, implying models " *
                "were not trained on the same sample. LOO-CV cannot be used."
            )
        )
    end

    return n_models, data_size
end
