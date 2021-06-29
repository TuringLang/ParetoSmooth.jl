module ImportanceSampling

using Base: AbstractFloat, Integer
using LoopVectorization
using Tullio

import ..ESS
import ..GPD

const LIKELY_ERROR_CAUSES = """
1. Bugs in the program that generated the sample, or otherwise incorrect input variables. 
2. Your chains failed to converge. Check your diagnostics. 
3. You do not have enough posterior samples (Less than ~100 samples) -- try sampling more values.
"""
const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to work
const SAMPLE_SOURCES = ["mcmc", "vi", "other"]

export Psis, psis

"""
    psis(log_ratios::AbstractArray{T:>AbstractFloat}; source::String="mcmc") -> Psis
Implements Pareto-smoothed importance sampling.

# Arguments
- `log_ratios::AbstractArray{T}`: An array of importance ratios on the log scale (for PSIS-LOO
these are *negative* log-likelihood values). Indices must be ordered as `[data, draw, chain]`:
`log_ratios[1, 2, 3]` should be the log-likelihood of the first data point, evaluated at the
second iteration of the third chain. Chain indices can be left off if there is only a single
chain. 
- `rel_eff::AbstractArray{T}`: An (optional) vector of relative effective sample sizes used
in ESS calculations. If left empty, calculated automatically using InferenceDiagnostics.jl's
default method. See `relative_eff` to calculate these values.
- `source::String="mcmc"`: A string or symbol describing the source of the sample being used.
If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be independent.
Currently permitted values are $SAMPLE_SOURCES.
"""
function psis(
    log_ratios::T, rel_eff::Union{Nothing, AbstractArray{F}}=nothing;
    source::Union{AbstractString,Symbol}="mcmc"
) where {F<:AbstractFloat,T<:AbstractArray{F,3}}

    source = lowercase(String(source))
    dimensions = size(log_ratios)

    numDataPoints = dimensions[1]
    posteriorSampleSize = dimensions[2] * dimensions[3]


    # Reshape to matrix (easier to deal with)
    log_ratios = reshape(log_ratios, numDataPoints, posteriorSampleSize)
    weights::AbstractArray{F} = similar(log_ratios)
    # Shift ratios by maximum to prevent overflow
    # then multiply by posterior sample size to avoid loss of precision for large samples
    @tturbo @. weights = posteriorSampleSize * exp(log_ratios - $maximum(log_ratios; dims=2))

    if rel_eff ≡ nothing
        if source == "mcmc"
            @info "Adjusting for autocorrelation. If the posterior samples are not " *
            "autocorrelated, specify the source of the posterior sample using the keyword " *
            "argument `source`. MCMC samples are always autocorrelated; VI samples are not."
            rel_eff = @tturbo ESS.relative_eff(inv.(reshape(weights, dimensions)))
        elseif source ∈ SAMPLE_SOURCES
            @info "Samples have not been adjusted for autocorrelation. If the posterior " * 
            "samples are autocorrelated, as in MCMC methods, ESS estimates will be " *
            "upward-biased, and standard error estimates will be downward-biased. " *
            "MCMC samples are always autocorrelated; VI samples are not."
            rel_eff = ones(size(log_ratios)[1])
        else 
            throw(ArgumentError("$source is not a valid source. " *
                "Valid sources are $SAMPLE_SOURCES."
                )
            )
        end
    end

    tailLength = similar(log_ratios, Int, numDataPoints)
    ξ = similar(log_ratios, F, numDataPoints)
    @tturbo @. tailLength = def_tail_length(posteriorSampleSize, rel_eff)
    @tturbo @. ξ = do_psis_i!($eachslice(weights; dims=1), tailLength)

    @tullio normConst[i] := weights[i, j]
    @tturbo weights .= weights ./ normConst
    ess = ESS.psis_n_eff(weights, rel_eff)

    weights = reshape(weights, dimensions)

    return Psis(
        weights,
        ξ,
        ess,
        tailLength,
        rel_eff,
        posteriorSampleSize,
        numDataPoints,
    )

end


function psis(log_ratios::AbstractMatrix{T}; kwargs...) where {T<:AbstractFloat}
    @info "Chain information was not provided; " * 
    "all samples are assumed to be drawn from a single chain."
    return psis(reshape(log_ratios, size(log_ratios), 1); kwargs...)
end

function psis(log_ratios::AbstractMatrix{T}, chain_index; kwargs...) where {T<:AbstractFloat}
    new_ratios = log_ratios 
    return psis(reshape(log_ratios, size(log_ratios), 1); kwargs...)
end


"""
    do_psis_i!(is_ratios::AbstractVector{AbstractFloat}, tail_length::Integer)::T

Do PSIS on a single vector, smoothing its tail values.

# Arguments

  - `is_ratios::AbstractVector{AbstractFloat}`: A vector of (not necessarily
  normalized) importance sampling ratios.

# Returns

  - `T<:AbstractFloat`: ξ, the shape parameter for the GPD; larger numbers indicate
  thicker tails.

# Extended help

Additional information can be found in the LOO package from R {Add link here}
{Add citation for Vehtari paper}
"""
function do_psis_i!(
    is_ratios::AbstractVector{T},
    tail_length::Integer,
) where {T<:AbstractFloat}
    len = length(is_ratios)
    ordering = sortperm(is_ratios; alg = QuickSort)
    permute!(is_ratios, ordering)

    # Define and check tail
    tailStartsAt = len - tail_length + 1  # index of smallest tail value
    @views tail = is_ratios[tailStartsAt:len]
    check_tail(tail)

    # Get value just before the tail starts:
    cutoff = is_ratios[tailStartsAt-1]
    ξ = psis_smooth_tail!(tail, cutoff)

    # truncate at max of raw wts; because is_ratios ∝ len / maximum, max(raw weights) = len
    clamp!(is_ratios, 0, len)
    # unsort the ratios to their original position:
    permute!(is_ratios, collect(1:len)[ordering])

    return ξ::T
end


"""
    def_tail_length(log_ratios::AbstractVector, rel_eff::AbstractFloat) -> tail_len::Integer

Define the tail length as in Vehtari et al. (2019). {Cite Vehtari paper here}
"""
function def_tail_length(length::I, rel_eff) where I<:Integer
    return I(ceil(min(length / 5, 3 * sqrt(length) / rel_eff)))
end



"""
    psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat} -> ξ::T

Takes an *already sorted* vector of observations from the tail and smooths it *in place*  
with PSIS before returning shape parameter `ξ`.
"""
function psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat}
    len = length(tail)
    @turbo @. tail = tail - cutoff

    # save time not sorting since x already sorted
    ξ, σ = GPD.gpdfit(tail)
    if ξ ≠ Inf
        @turbo @. tail = GPD.gpd_quantile($(1:len) / (len + 1), ξ, σ) + cutoff
    end
    return ξ
end



##########################
#####  PSIS STRUCTS  #####
##########################


"""
    Psis{V<:AbstractVector{F},I<:Integer} where {F<:AbstractFloat}

A struct containing the results of Pareto-smoothed improtance sampling.

# Fields
- `weights`: A vector of smoothed, truncated, and *normalized* importance sampling weights.
- `pareto_k`: Estimates of the shape parameter ``k`` of the generalized Pareto distribution.
- `ess`: Estimated effective sample size for each LOO evaluation.
- `tail_len`: Vector of tail lengths used for smoothing the generalized Pareto distribution.
- `dims`: Named tuple of length 2 containing `s` (posterior sample size) and `n` (number of
observations).
"""
struct Psis{
    F<:AbstractFloat,
    AF<:AbstractArray{F},
    VF<:AbstractVector{F},
    I<:Integer,
    VI<:AbstractVector{I},
}
    weights::AF
    pareto_k::VF
    ess::VF
    tail_len::VI
    rel_eff::VF
    posterior_sample_size::I
    data_size::I
end



##########################
#### HELPER FUNCTIONS ####
##########################

function check_input_validity_psis(
    log_ratios::AbstractArray{T,3},
    rel_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    if any(isnan, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN values)."))
    elseif any(isinf, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    elseif any(isnan, rel_eff)
        throw(ArgumentError("Invalid input for `rel_eff` (contains NaN values)."))
    elseif any(isinf, rel_eff)
        throw(DomainError("Invalid input for `rel_eff` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `rel_eff` (array is empty)."))
    elseif length(rel_eff) ≠ size(log_ratios, 1)
        throw(ArgumentError("Size of `rel_eff` does not equal the number of data points."))
    end
    return nothing
end


"""
Check the tail to make sure a GPD fit is possible.
"""
function check_tail(tail::AbstractVector{T}) where {T<:AbstractFloat}
    if maximum(tail) ≈ minimum(tail)
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: all tail values are the same. \n$LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) < MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: tail length was too short.\n$LIKELY_ERROR_CAUSES",
            ),
        )
    end
    return nothing
end

end