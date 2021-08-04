import Base.show


"""

# LooCompare

A struct containing the results of a PsisLoo model comparison.

$(FIELDS)

# Extended help

# Fields
  - `psis::Vector{PsisLoo}` : Vector of PsisLoo objects.
  - `table::KeyedArray` : Comparison table.

# Example of a comparison table
```
┌───────┬───────────┬─────────┬────────┐
│       │ elpd_diff │ se_diff │ weight │
├───────┼───────────┼─────────┼────────┤
│ m5_1t │      0.00 │    0.00 │   0.67 │
│ m5_3t │     -0.69 │    0.42 │   0.33 │
│ m5_2t │     -6.68 │    4.74 │   0.00 │
└───────┴───────────┴─────────┴────────┘
```

where:

1. `elpd_diff`  : Difference between total loo_est values between models.
2. `se_diff`    : Standard error of the difference in total l00_est values.
3. `weight`     : Relative support for each model.

In this example table the models have been sorted in ascending total loo_est values.
The PsisLoo objects in the field `psis` is sorted as listed in `table`.

See also: [`PsisLoo`](@ref).
"""
struct LooCompare
    psis::Vector{PsisLoo}
    table::KeyedArray
end

"""

# loo_compare

Construct a PsisLoo comparison table for loglikelihood matrices.

$(SIGNATURES)

# Extended help

### Required arguments
    - `loglikelihoods::Vector{Array{Float64, 3}}` : Vector of loglikelihood matrices

### Optional arguments
    - `model_names=nothing` : Optional specify models
    - `sort_models=true` : Sort models according to ascending elpd values

### Return values
    - `result::LooCompare` : LooCompare object

See also: [`LooCompare`](@ref).
"""
function loo_compare(
    loglikelihoods::Vector{Array{Float64, 3}};
    model_names=nothing, 
    sort_models=true)

    nmodels = length(loglikelihoods)

    if isnothing(model_names)
        mnames = ["model_$i" for i in 1:nmodels]
    else
        mnames = model_names
    end

    psis_array = psis_loo.(loglikelihoods)
    println(typeof(psis_array))
    loo_compare(psis_array; model_names, sort_models)
end

function loo_compare(
    psis::Vector{PsisLoo};
    model_names=nothing, 
    sort_models=true)

    psis_values = Vector{Float64}(undef, nmodels)
    se_values = Vector{Float64}(undef, nmodels)
    loos = Vector{Vector{Float64}}(undef, nmodels)

    nmodels = length(psis)

    for i in 1:nmodels
        psis_values[i] = psis[i].estimates(:loo_est, :total)
        se_values[i] = psis[i].estimates(:loo_est, :se_total)
        loos[i] = psis[i].pointwise(:loo_est)
    end

    if sort_models
        ind = sortperm([psis_values[i][1] for i in 1:nmodels]; rev=true)
        psis = psis[ind]
        psis_values = psis_values[ind]
        se_values = se_values[ind]
        loos = loos[ind]
        mnames = mnames[ind]
    end

    # Setup comparison vectors

    elpd_diff = zeros(nmodels)
    se_diff = zeros(nmodels)
    weight = ones(nmodels)

    # Compute comparison values

    for i in 2:nmodels
        elpd_diff[i] = psis_values[i] - psis_values[1]
        diff = loos[1] - loos[i]
        se_diff[i] = √(length(loos[i]) * var(diff; corrected=false))
    end
    data = elpd_diff
    data = hcat(data, se_diff)

    sumval = sum([exp(psis_values[i]) for i in 1:nmodels])
    @. weight = exp(psis_values) / sumval
    data = hcat(data, weight)
    
    # Create KeyedArray object

    table = KeyedArray(
        data,
        model = mnames,
        statistic = [:elpd_diff, :se_diff, :weight],
    )

    # Return LooCompare object
    
    LooCompare(psis, table)

end

function Base.show(io::IO, ::MIME"text/plain", loo_compare::LooCompare)
    table = loo_compare.table
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.model,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end

export
    LooCompare,
    loo_compare
