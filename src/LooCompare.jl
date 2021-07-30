import Base.show


"""

# LooCompare

A struct containing the results of PsisLoo model comparisom.

$(FIELDS)

# Extended help

# Fields
  - `psis::Vector{PsisLoo}`                      : A vector of PsisLoo objects.
  - `table::KeyedArray`                          : Comparison table.

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
```julia
* `loglikelihoods::Vector{Array{Float64, 3}}`    : Vector of loglikelihood matrices
```

### Optional arguments
```julia
* `model_names=nothing`                          : Optional specify models
* `sort_models=true`                             : Sort models
```

### Return values
```julia
* `result::LooCompare`                           : LooCompare object
```

"""
function loo_compare(
    loglikelihoods::Vector{Array{Float64, 3}};
    model_names=Union{Nothing, Vector{AbstractString}}, 
    sort_models=true)

    nmodels = length(loglikelihoods)

    if isnothing(model_names)
        mnames = ["model_$i" for i in 1:nmodels]
    else
        mnames = model_names
    end

    psis = Vector{PsisLoo}(undef, nmodels)
    psis_values = Vector{Float64}(undef, nmodels)
    se_values = Vector{Float64}(undef, nmodels)
    loos = Vector{Vector{Float64}}(undef, nmodels)

    for i in 1:nmodels
        psis[i] = psis_loo(loglikelihoods[i])
        psis_values[i] = psis[i].estimates(:loo_est, :total)
        se_values[i] = psis[i].estimates(:loo_est, :se_total)
        loos[i] = psis[i].pointwise(:loo_est)
    end

    if sort_models
        ind = sortperm([psis_values[i][1] for i in 1:nmodels]; rev=true)
        psis_values = psis_values[ind]
        se_values = se_values[ind]
        loos = loos[ind]
        mnames = mnames[ind]
    end

    # Setup comparison vectors

    dpsis = zeros(nmodels)
    dse = zeros(nmodels)
    weights = ones(nmodels)

    # Compute comparison values

    for i in 2:nmodels
        dpsis[i] = psis_values[i] - psis_values[1]
        diff = loos[1] - loos[i]
        dse[i] = √(length(loos[i]) * var(diff; corrected=false))
    end
    data=dpsis
    data = hcat(data, dse)

    sumval = sum([exp(psis_values[i]) for i in 1:nmodels])
    for i in 1:nmodels
        weights[i] = exp(psis_values[i])/sumval
    end
    data = hcat(data, weights)
    
    # Create KeyedArray object

    table = KeyedArray(
        data,
        model = mnames,
        statistic = [:d_PSIS, :d_SE, :weight],
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
