
using AxisKeys
using PrettyTables
export PsisLoo, AbstractLoo, AbstractLooMethod, PsisLooMethod

abstract type AbstractLoo end

"""
    PsisLoo{
        F <: AbstractFloat,
        AF <: AbstractArray{F},
        VF <: AbstractVector{F},
        I <: Integer,
        VI <: AbstractVector{I},
    } <: AbstractLoo

A struct containing the results of leave-one-out cross validation using Pareto smoothed
importance sampling.

# Fields

  - `estimates::KeyedArray`: A `KeyedArray` with two columns (`:Estimate`, `:SE`) and three
    rows (`:total_score`, `:overfit`, `:avg_score`). This contains point estimates and
    standard errors for the total log score (the sum of all errors); the effective number of 
    parameters (difference between in-sample and out-of-sample predictive accuracy); and the
    average log-score (Sometimes referred to as the ELPD). See the extended help for more
    details.
  - `pointwise::KeyedArray`: An array of pointwise values
  - `psis_object::Psis`: A `Psis` object containing the results of Pareto-smoothed 
    importance sampling.


# Extended help

The total score depends on the sample size, and summarizes the weight of evidence for or
against a model. Total scores are on an interval scale, meaning that only differences of
scores are meaningful. *It is not possible to interpret a total score by looking at it.*
The total score is not a relative goodness-of-fit statistic (for this, see the average
score).


The overfit is equal to the difference between the in-sample and out-of-sample predictive
accuracy. When using the log probability score, it is equal to the "effective number of
parameters" -- a model with an overfit of 2 has "about as much overfit" as a model with 2
free parameters and flat priors.


The average score is the total score, divided by the sample size. It estimates the expected
log score, i.e. the expectation of the log probability density of observing the next point.
The average score is a relative goodness-of-fit statistic which does not depend on sample
size. 


Unlike with chi-square goodness of fit tests, models do not have to be nested for PSIS-LOO.


See also: [`psis_loo`]@ref, [`Psis`]@ref

"""
struct PsisLoo{
    F <: AbstractFloat,
    AF <: AbstractArray{F},
    VF <: AbstractVector{F},
    I <: Integer,
    VI <: AbstractVector{I},
} <: AbstractLoo
    estimates::KeyedArray
    pointwise::KeyedArray
    psis_object::Psis{F, AF, VF, I, VI}
end

abstract type AbstractLooMethod end

struct PsisLooMethod <: AbstractLooMethod end

function _throw_pareto_k_warning(ξ)
    if any(ξ .≥ .7)
        @warn "Some Pareto k values are very high (>0.7), indicating that PSIS has " * 
        "failed to approximate the true distribution."
    elseif any(ξ .≥ .5)
        @info "Some Pareto k values are slightly high (>0.5); some pointwise estimates " *
        "may be slow to converge or have high variance."
    end
end


function Base.show(io::IO, ::MIME"text/plain", loo_object::PsisLoo)
    table = loo_object.estimates
    _throw_pareto_k_warning(loo_object.pointwise(:pareto_k))
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.criterion,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end
