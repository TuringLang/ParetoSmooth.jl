
using AxisKeys
using PrettyTables

export AbstractCVMethod, AbstractCV

const POINTWISE_LABELS = (:cv_est, :naive_est, :p_eff, :ess, :pareto_k)
const CV_DESC = """
# Fields

  - `estimates::KeyedArray`: A KeyedArray with columns `:total, :se_total, :mean, :se_mean`,
    and rows `:cv_est, :naive_est, :p_eff`. See `# Extended help` for more.
      - `:cv_est` contains estimates for the out-of-sample prediction error, as
        estimated using leave-one-out cross validation.
      - `:naive_est` contains estimates of the in-sample prediction error.
      - `:p_eff` is the effective number of parameters -- a model with a `p_eff` of 2 is 
        "about as overfit" as a model with 2 parameters and no regularization. It equals the 
        difference between the previous two estimators, and measures how much your model
        tends to overfit the data.
  - `pointwise::KeyedArray`: A `KeyedArray` of pointwise estimates with 5 columns --
      - `:cv_est` contains the estimated out-of-sample error for this point, as measured
      using leave-one-out cross validation.
      - `:naive_est` contains the in-sample estimate of error for this point.
      - `:p_eff` is the difference in the two previous estimates.
      - `:ess` is the effective sample size, which estimates the simulation error caused by 
        using Monte Carlo estimates. It does not measure the accuracy of predictions.  
      - `:pareto_k` is the estimated value for the parameter `ξ` of the generalized Pareto
        distribution. Values above .7 indicate that PSIS has failed to approximate the true
        distribution.
  - `psis_object::Psis`: A `Psis` object containing the results of Pareto-smoothed 
    importance sampling.
  - `mcse`: A float containing the estimated Monte Carlo standard error for the total 
    cross-validation estimate.


# Extended help

The total score depends on the sample size, and summarizes the weight of evidence for or
against a model. Total scores are on an interval scale, meaning that only differences of
scores are meaningful. *It is not possible to interpret a total score by looking at it.*
The total score is not a goodness-of-fit statistic (for this, see the average score).


The average score is the total score, divided by the sample size. It estimates the expected
log score, i.e. the expectation of the log probability density of observing the next point.
The average score is a relative goodness-of-fit statistic which does not depend on sample
size. 


Unlike for chi-square goodness of fit tests, models do not have to be nested for model
comparison using cross-validation methods.
"""


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
### BAYESIAN BOOTSTRAP ###
##########################

# """
#     BayesCV{
#         F <: Real,
#         AF <: AbstractArray{F},
#         VF <: AbstractVector{F},
#         I <: Integer,
#         VI <: AbstractVector{I},
#     } <: AbstractCV

# A struct containing the results of cross-validation using the Bayesian bootstrap.

# $CV_DESC

# See also: [`bayes_cv`]@ref, [`psis_loo`]@ref, [`psis`]@ref, [`Psis`]@ref
# """
# struct BayesCV{
#     F <: Real,
#     AF <: AbstractArray{F},
#     VF <: AbstractVector{F},
#     I <: Integer,
#     VI <: AbstractVector{I},
# } <: AbstractCV
#     estimates::KeyedArray
#     posteriors::KeyedArray
#     psis_object::Psis{F, AF, VF, I, VI}
# end


# function Base.show(io::IO, ::MIME"text/plain", cv_object::BayesCV)
#     table = cv_object.estimates
#     post_samples = cv_object.psis_object.posterior_sample_size
#     data_size = cv_object.psis_object.data_size
#     _throw_pareto_k_warning(cv_object.resamples(:pareto_k))
#     println(
#         "Results of Bayesian bootstrap CV with $post_samples Monte Carlo samples and " *
#         "$data_size data points.",
#     )
#     return pretty_table(
#         table;
#         compact_printing=false,
#         header=table.statistic,
#         row_names=table.criterion,
#         formatters=ft_printf("%5.2f"),
#         alignment=:r,
#     )
# end
