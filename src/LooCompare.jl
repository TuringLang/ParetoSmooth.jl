import Base.show


"""

# LooCompare

A struct containing the results of a PsisLoo model comparison.

$(FIELDS)

# Extended help

### Fields
  - `psis::Vector{PsisLoo}` : Vector of PsisLoo objects.
  - `table::KeyedArray` : Comparison table.

#### Example of a comparison table
```
┌───────┬────────────────┬───────────────────┬────────┐
│       │ loo_score_diff │ se_loo_score_diff │ weight │
├───────┼────────────────┼───────────────────┼────────┤
│ m5_1t │           0.00 │              0.00 │   0.67 │
│ m5_3t │          -0.69 │              0.42 │   0.33 │
│ m5_2t │          -6.68 │              4.74 │   0.00 │
└───────┴────────────────┴───────────────────┴────────┘
```

where:

1. `loo_score_diff` : Difference in loo_scores between models.
2. `se_loo_score_diff` : Standard error of the difference in loo_scores between models.
3. `weight` : Relative support for each model.

The `loo_score` is the sum (`total`) of the `loo_est` values in the PsisLoo object.

In this example table the models have been sorted in ascending loo_score values.
The PsisLoo objects in the field `psis` are sorted as listed in `table`.

See also: [`PsisLoo`}(@ref) 
"""
struct LooCompare
    psis::Vector{PsisLoo}
    table::KeyedArray
end

"""
Construct a PsisLoo comparison table from a vector of loglikelihood matrices.
Return a LooCompare object.

$(SIGNATURES)

# Extended help

### Required arguments
    - `loglikelihoods::Vector{Array{AF, 3}} where {AF <: AbstractFloat}` : Vector of loglikelihood matrices

### Optional arguments
    - `model_names=nothing` : A vector of model names
    - `sort_models=true` : Sort models according to ascending loo_score values

### Return values
    - `result::LooCompare` : LooCompare object

See also: [`LooCompare`](@ref).
"""
function loo_compare(
    loglikelihood_vector::Vector{Array{AF, 3}} where {AF <: Real};
    model_names=nothing, 
    sort_models=true)

    nmodels = length(loglikelihood_vector)

    if isnothing(model_names)
        mnames = ["model_$i" for i in 1:nmodels]
    else
        mnames = model_names
    end

    psis_array = psis_loo.(loglikelihood_vector)
    return loo_compare(psis_array; model_names, sort_models)
end

"""
Construct a PsisLoo comparison table from a NamedTuple.
The keys of the NamedTuple are used as model names,
the values must be a subtype of PsisLoo.
Return a LooCompare object.


$(SIGNATURES)

# Extended help

### Required arguments
    - `nt::NamedTuple` : NamedTuple

### Optional arguments
    - `sort_models=true` : Sort models according to ascending loo_score values

### Return values
    - `result::LooCompare` : LooCompare object


See also: [`LooCompare`](@ref).
"""
function loo_compare(
    nt::NamedTuple;
    model_names=nothing, 
    sort_models=true)

    nmodels = length(keys(nt))

    if !(eltype(nt) <: PsisLoo) 
        throw(ArgumentError("Not a NamedTuple with PsisLoo type values."))
    end

    mnames = [Symbol(keys(nt)[i]) for i in 1:length(values(nt))]
    psis_array = [values(nt)[i] for i in 1:length(values(nt))]
    
    return loo_compare(psis_array; model_names=mnames, sort_models)
end

"""
Construct a PsisLoo comparison table from a vector of PsisLoo objects.
Return a LooCompare object.

$(SIGNATURES)

# Extended help

### Required arguments
    - `psis_vector::Vector{PsisLoo{AF, Array{AF, 3}, Vector{AF}, I, Vector{I}}} where {AF <: AbstractFloat, I <: Integer}` : Vector of loglikelihood matrices

### Optional arguments
    - `model_names=nothing` : A vector of model names
    - `sort_models=true` : Sort models according to ascending loo_score values

### Return values
    - `result::LooCompare` : LooCompare object

See also: [`LooCompare`](@ref).
"""
function loo_compare(
    psis_vector::Vector{PsisLoo{AF, Array{AF, 3}, Vector{AF}, I, Vector{I}}}
        where {AF <: Real, I <: Integer};
    model_names=nothing, 
    sort_models=true)

    # Deepcopy because we might reorder psis_vector.

    psis = deepcopy(psis_vector)
    nmodels = length(psis)

    if isnothing(model_names)
        mnames = ["model_$i" for i in 1:nmodels]
    else
        mnames = model_names
    end

    # Extract relevant values from PsisLoo objects.

    psis_values = [psis[i].estimates(:loo_est, :total) for i in 1:nmodels]
    se_values = [psis[i].estimates(:loo_est, :se_total) for i in 1:nmodels]
    loos = [psis[i].pointwise(:loo_est) for i in 1:nmodels]

    if sort_models
        ind = sortperm([psis_values[i][1] for i in 1:nmodels]; rev=true)
        psis = psis[ind]
        psis_values = psis_values[ind]
        se_values = se_values[ind]
        loos = loos[ind]
        mnames = mnames[ind]
    end

    # Compute differences between models.

    loo_score_diff = [psis_values[i] - psis_values[1] for i in 1:nmodels]
    se_loo_score_diff = 
        [√(length(loos[i]) * var(loos[1] - loos[i]; corrected=false)) 
            for i in 1:nmodels]

    data = loo_score_diff
    data = hcat(data, se_loo_score_diff)

    sumval = sum([exp(psis_values[i]) for i in 1:nmodels])
    weight = [exp(psis_values[i]) / sumval for i in 1:nmodels]
    data = hcat(data, weight)
    
    # Create KeyedArray object

    table = KeyedArray(
        data,
        model = mnames,
        statistic = [:loo_score_diff, :se_loo_score_diff, :weight],
    )

    # Return LooCompare object
    
    return LooCompare(psis, table)

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
