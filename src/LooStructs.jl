
using AxisKeys
using PrettyTables
export PsisLoo, PsisLooMethod, Psis, BayesCV

const POINTWISE_LABELS = (:cv_est, :naive_est, :overfit, :ess, :pareto_k)
const CV_DESC = """
# Fields

  - `estimates::KeyedArray`: A KeyedArray with columns `:total, :se_total, :mean, :se_mean`,
    and rows `:cv_est, :naive_est, :overfit`. See `# Extended help` for more.
      - `:cv_est` contains estimates for the out-of-sample prediction error, as
        predicted using the jackknife (LOO-CV).
      - `:naive_est` contains estimates of the in-sample prediction error.
      - `:overfit` is the difference between the previous two estimators, and estimates 
        the amount of overfitting. When using the log probability score, it is equal to 
        the effective number of parameters -- a model with an overfit of 2 is "about as
        overfit" as a model with 2 independent parameters that have a flat prior.
  - `pointwise::KeyedArray`: A `KeyedArray` of pointwise estimates with 5 columns --
      - `:cv_est` contains the estimated out-of-sample error for this point, as measured
        using leave-one-out cross validation.
      - `:naive_est` contains the in-sample estimate of error for this point.
      - `:overfit` is the difference in the two previous estimates.
      - `:ess` is the effective sample size, which measures the simulation error caused by 
        using Monte Carlo estimates. It is *not* related to the actual sample size, and it
        does not measure how accurate your predictions are.     
    - `:pareto_k` is the estimated value for the parameter `ξ` of the generalized Pareto
      distribution. Values above .7 indicate that PSIS has failed to approximate the true
      distribution.
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
parameters" -- a model with an overfit of 2 is "about as overfit" as a model with 2
free parameters and flat priors.


The average score is the total score, divided by the sample size. It estimates the expected
log score, i.e. the expectation of the log probability density of observing the next point.
The average score is a relative goodness-of-fit statistic which does not depend on sample
size. 


Unlike for chi-square goodness of fit tests, models do not have to be nested for model
comparison using cross-validation methods.
"""


###########################
### IMPORTANCE SAMPLING ###
###########################

"""
    Psis{V<:AbstractVector{F},I<:Integer} where {F<:Real}

A struct containing the results of Pareto-smoothed importance sampling.

# Fields
  - `weights`: A vector of smoothed, truncated, and normalized importance sampling weights.
  - `pareto_k`: Estimates of the shape parameter `k` of the generalized Pareto distribution.
  - `ess`: Estimated effective sample size for each LOO evaluation.
  - `tail_len`: Vector indicating how large the "tail" is for each observation.
  - `dims`: Named tuple of length 2 containing `s` (posterior sample size) and `n` (number
    of observations).
"""
struct Psis{
    F<:Real,
    AF<:AbstractArray{F,3},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
}
    weights::AF
    pareto_k::VF
    ess::VF
    r_eff::VF
    tail_len::VI
    posterior_sample_size::I
    data_size::I
end


function _throw_pareto_k_warning(ξ)
    if any(ξ .≥ .7)
        @warn "Some Pareto k values are very high (>0.7), indicating that PSIS has " * 
        "failed to approximate the true distribution."
    elseif any(ξ .≥ .5)
        @info "Some Pareto k values are slightly high (>0.5); some pointwise estimates " *
        "may be slow to converge or have high variance."
    end
end


function Base.show(io::IO, ::MIME"text/plain", psis_object::Psis)
    table = hcat(psis_object.pareto_k, psis_object.ess)
    post_samples = psis_object.posterior_sample_size
    data_size = psis_object.data_size
    println("Results of PSIS with $post_samples Monte Carlo samples and " *
    "$data_size data points.")
    _throw_pareto_k_warning(psis_object.pareto_k)
    return pretty_table(
        table;
        compact_printing=false,
        header=[:pareto_k, :ess],
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end



##########################
#### CROSS VALIDATION ####
##########################

"""
    AbstractCV
An abstract type used in cross-validation.
"""
abstract type AbstractCV end

"""
    AbstractCVMethod
An abstract type used to dispatch the correct method for cross validation.
"""
abstract type AbstractCVMethod end



##########################
######## PSIS-LOO ########
##########################

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
end




function Base.show(io::IO, ::MIME"text/plain", loo_object::PsisLoo)
    table = loo_object.estimates
    _throw_pareto_k_warning(loo_object.pointwise(:pareto_k))
    post_samples = loo_object.psis_object.posterior_sample_size
    data_size = loo_object.psis_object.data_size
    println("Results of PSIS-LOO-CV with $post_samples Monte Carlo samples and " *
    "$data_size data points.")
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.criterion,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end



##########################
### BAYESIAN BOOTSTRAP ###
##########################

"""
    BayesCV{
        F <: Real,
        AF <: AbstractArray{F},
        VF <: AbstractVector{F},
        I <: Integer,
        VI <: AbstractVector{I},
    } <: AbstractCV

A struct containing the results of cross-validation using the Bayesian bootstrap.

$CV_DESC

See also: [`bayes_cv`]@ref, [`psis_loo`]@ref, [`psis`]@ref, [`Psis`]@ref
"""
struct BayesCV{
    F <: Real,
    AF <: AbstractArray{F},
    VF <: AbstractVector{F},
    I <: Integer,
    VI <: AbstractVector{I},
} <: AbstractCV
    estimates::KeyedArray
    posteriors::KeyedArray
    psis_object::Psis{F, AF, VF, I, VI}
end


function Base.show(io::IO, ::MIME"text/plain", cv_object::BayesCV)
    table = cv_object.estimates
    post_samples = cv_object.psis_object.posterior_sample_size
    data_size = cv_object.psis_object.data_size
    _throw_pareto_k_warning(cv_object.resamples(:pareto_k))
    println("Results of Bayesian bootstrap CV with $post_samples Monte Carlo samples and " *
    "$data_size data points.")
    return pretty_table(
        table;
        compact_printing=false,
        header=table.statistic,
        row_names=table.criterion,
        formatters=ft_printf("%5.2f"),
        alignment=:r,
    )
end
