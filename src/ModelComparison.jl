import Base.show

export loo_compare


"""
    ModelComparison

A struct containing the results of model comparison.

# Fields

  - `pointwise::KeyedArray`:
  - `estimates::KeyedArray`: A table containing the results of model comparison, with the
    following columns --
      + `cv_est`: The difference in *total* leave-one-out cross validation scores
        between models.
      + `se_loo_diff`: The standard error for each row of `cv_est`.
      + `weight`: A set of Akaike-like weights assigned to each model, which can be used in
        pseudo-Bayesian model averaging.

# Example

```
┌───────┬──────────┬───────┬────────┐
│       │   cv_est │ se_cv │ weight │
├───────┼──────────┼───────┼────────┤
│ m5_1t │     0.00 │  0.00 │   0.67 │
│ m5_3t │    -0.69 │  0.42 │   0.33 │
│ m5_2t │    -6.68 │  4.74 │   0.00 │
└───────┴──────────┴───────┴────────┘
```

See also: [`PsisLoo`](@ref)
"""
struct ModelComparison
    pointwise::KeyedArray
    estimates::KeyedArray
end



"""
    function loo_compare(
        cv_results::PsisLoo...;
        sort_models::Bool=true,
        [, model_names::Tuple{Symbol}]
    ) -> ModelComparison

Construct a model comparison table from several [`PsisLoo`](@ref) objects.

# Arguments

    - `cv_results`: One or more [`PsisLoo`](@ref) objects to be compared. Alternatively,
    a tuple or named tuple of `PsisLoo` objects can be passed. If a named tuple is passed,
    these names will be used to label each model. 
    - `model_names`: A vector or tuple of strings or symbols used to identify models. If
    none, models are numbered using the order of the arguments.
    - `sort_models=true`: Sort models by total score (best first).

See also: [`ModelComparison`](@ref), [`PsisLoo`](@ref), [`psis_loo`](@ref)
"""
function loo_compare(
    cv_results::PsisLoo...;
    model_names::Stringlike=[Symbol("model_$i") for i in 1:n_models],
    sort_models::Bool=true,
) where {Stringlike <: Base.AbstractVecOrTuple{<:Union{AbstractString, Symbol}}}

    model_names = Symbol.(model_names)
    model_names = [model for model in model_names]
    nmodels = length(cv_results)
    if nmodels != length(model_names)
        throw(ArgumentError("Number of model names does not match size of `cv_results`."))
    end

    # Extract relevant values from PsisLoo objects

    estimates = [cv_results[i].estimates(:cv_est, :total) for i in 1:nmodels]
    pointwise = [cv_results[i].pointwise for i in 1:nmodels]
    cv_pointwise = [cv_results[i].pointwise(:cv_est) for i in 1:nmodels]

    if sort_models
        ind = sortperm([estimates[i][1] for i in 1:nmodels]; rev=true)
        cv_results = cv_results[ind]
        estimates = estimates[ind]
        pointwise = pointwise[ind]
        cv_pointwise = cv_pointwise[ind]
        model_names = model_names[ind]
    end

    # Compute differences between models
    cv_est = [estimates[i] - estimates[1] for i in 1:nmodels]
    se_cv_est = [
        sqrt(length(cv_pointwise[1]) * var(cv_pointwise[1] - cv_pointwise[i])) for
        i in 1:nmodels
    ]

    table = cv_est
    table = hcat(table, se_cv_est)

    sumval = sum([exp(estimates[i]) for i in 1:nmodels])
    weight = [exp(estimates[i]) / sumval for i in 1:nmodels]
    table = hcat(table, weight)

    table = wrapdims(table; model=model_names, statistic=[:cv_est, :se_cv_est, :weight])

    # Construct pointwise differences

    pointwise = cat(pointwise...; dims=3)
    pointwise = KeyedArray(
        pointwise;
        data=1:size(pointwise, :data),
        statistic=[:cv_est, :naive_est, :p_eff, :mcse, :pareto_k],
        model=model_names,
    )

    # Subtract the effective number of params and elpd ests; leave mcse+pareto_k the same
    base_case = pointwise[data=:, statistic=1:3, model=1]
    @inbounds @simd for model_number in axes(pointwise, :model)
        @. pointwise[:, 1:3, model_number] = pointwise[:, 1:3, model_number] - base_case
    end

    return ModelComparison(pointwise, table)

end


function loo_compare(cv_results::NamedTuple; kwargs...)
    return loo_compare(values(cv_results); model_names=keys(cv_results), kwargs...)
end


function loo_compare(cv_results::Base.AbstractVecOrTuple{<:PsisLoo}; kwargs...)
    return loo_compare(cv_results...; kwargs...)
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

