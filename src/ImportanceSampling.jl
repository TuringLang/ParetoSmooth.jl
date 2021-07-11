using LoopVectorization
using Tullio


const LIKELY_ERROR_CAUSES = """
1. Bugs in the program that generated the sample, or otherwise incorrect input variables. 
2. Your chains failed to converge. Check your diagnostics. 
3. You do not have enough posterior samples (Less than ~100 samples) -- try sampling more values.
"""
const MIN_TAIL_LEN = 16  # Minimum size of a tail for PSIS to give sensible answers
const SAMPLE_SOURCES = ["mcmc", "vi", "other"]

export Psis, psis

"""
    psis(
        log_ratios::AbstractArray{T:>AbstractFloat}, 
        r_eff; 
        source::String="mcmc", 
        log_weights::Bool=false
        ) -> Psis

Implements Pareto-smoothed importance sampling (PSIS).

# Arguments
## Positional Arguments
- `log_ratios::AbstractArray{T}`: An array of importance ratios on the log scale (for 
PSIS-LOO these are *negative* log-likelihood values). Indices must be ordered as 
`[data, draw, chain]`: `log_ratios[1, 2, 3]` should be the log-likelihood of the first data 
point, evaluated at the second iteration of the third chain. Chain indices can be left off 
if there is only a single chain, or if keyword argument `chain_index` is provided.
- `r_eff::AbstractArray{T}`: An (optional) vector of relative effective sample sizes used 
in ESS calculations. If left empty, calculated automatically using the FFTESS method 
from InferenceDiagnostics.jl. See `relative_eff` to calculate these values. 

## Keyword Arguments

- `chain_index::Vector{Integer}`: An (optional) vector of integers indicating which chain 
each sample belongs to.
- `source::String="mcmc"`: A string or symbol describing the source of the sample being 
used. If `"mcmc"`, adjusts ESS for autocorrelation. Otherwise, samples are assumed to be 
independent. Currently permitted values are $SAMPLE_SOURCES.
- `log_weights::Bool=false`: Return the logarithm of the weights instead of the weights themselves. 
"""
function psis(
    log_ratios::T, r_eff::AbstractArray{F}=similar(log_ratios,0);
    source::Union{AbstractString,Symbol}="mcmc", log_weights::Bool=false
) where {F<:AbstractFloat,T<:AbstractArray{F,3}}

    source = lowercase(String(source))
    dims = size(log_ratios)

    data_size = dims[1]
    post_sample_size = dims[2] * dims[3]


    # Reshape to matrix (easier to deal with)
    log_ratios = reshape(log_ratios, data_size, post_sample_size)
    weights::AbstractArray{F} = similar(log_ratios)
    # Shift ratios by maximum to prevent overflow
    @tturbo @. weights = exp(log_ratios - $maximum(log_ratios; dims=2))

    r_eff = _generate_r_eff(weights, dims, r_eff, source)
    check_input_validity_psis(reshape(log_ratios, dims), r_eff)
    

    tail_length = similar(log_ratios, Int, data_size)
    ξ = similar(log_ratios, F, data_size)
    @tturbo @. tail_length = def_tail_length(post_sample_size, r_eff)
    @tturbo @. ξ = do_psis_i!($eachrow(weights), tail_length)

    @tullio norm_const[i] := weights[i, j]
    @tturbo @. weights /= norm_const
    ess = psis_n_eff(weights, r_eff)

    weights = reshape(weights, dims)
    
    if log_weights
        @tturbo @. weights = log(weights)
    end

    return Psis(
        weights,
        ξ,
        ess,
        r_eff,
        tail_length,
        post_sample_size,
        data_size,
    )

end


function psis(log_ratios::AbstractMatrix{T}, 
    r_eff::AbstractVector{T}=similar(log_ratios, 0); 
    chain_index::AbstractVector{I}=assume_one_chain(log_ratios), 
    kwargs...
    ) where {T<:AbstractFloat, I<:Integer}

    indices = unique(chain_index)
    biggest_idx = maximum(indices)
    dims = size(log_ratios)
    if dims[2] ≠ length(chain_index)
        throw(ArgumentError("Some entries do not have a chain index."))
    elseif !issetequal(indices, 1:biggest_idx)
        throw(ArgumentError("Indices must be numbered from 1 through the total number of chains."))
    else
        # Check how many elements are in each chain, assign to "counts"
        counts = count.(eachslice(chain_index .== indices'; dims=2))
        # check if all inputs are the same length
        if !all(==(counts[1]), counts)
            throw(ArgumentError("All chains must be of equal length."))
        end
    end
    new_ratios = similar(log_ratios, dims[1], dims[2] ÷ biggest_idx, biggest_idx)
    for i in 1:biggest_idx    
        new_ratios[:, :, i] .= log_ratios[:, chain_index .== i]
    end

    return psis(new_ratios, r_eff; kwargs...)
end


"""
    do_psis_i!(is_ratios::AbstractVector{AbstractFloat}, tail_length::Integer)::T

Do PSIS on a single vector, smoothing its tail values.

# Arguments

- `is_ratios::AbstractVector{AbstractFloat}`: A vector of importance sampling ratios, 
scaled to have a maximum of 1.

# Returns

- `T<:AbstractFloat`: ξ, the shape parameter for the GPD; big numbers indicate thick tails.

# Extended help

Additional information can be found in the LOO package from R.
"""
function do_psis_i!(
    is_ratios::AbstractVector{T},
    tail_length::Integer,
) where {T<:AbstractFloat}
    len = length(is_ratios)

    # sort is_ratios and also get results of sortperm() at the same time
    ordering = sortperm(is_ratios; alg=QuickSort)
    sorted_ratios = is_ratios[ordering]

    # Define and check tail
    tail_start = len - tail_length + 1  # index of smallest tail value
    @views tail = sorted_ratios[tail_start:len]
    check_tail(tail)

    # Get value just before the tail starts:
    cutoff = sorted_ratios[tail_start-1]
    ξ = psis_smooth_tail!(tail, cutoff)

    # truncate at max of raw weights (1 after scaling)
    clamp!(sorted_ratios, 0, 1)
    # unsort the ratios to their original position:
    is_ratios .= @views sorted_ratios[invperm(ordering)]
    
    return ξ::T
end


"""
    def_tail_length(log_ratios::AbstractVector, r_eff::AbstractFloat) -> tail_len::Integer

Define the tail length as in Vehtari et al. (2019).
"""
function def_tail_length(length::I, r_eff) where {I<:Integer}
    return I(ceil(min(length / 5, 3 * sqrt(length / r_eff))))
end



"""
    psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat} -> ξ::T

Takes an *already sorted* vector of observations from the tail and smooths it *in place*  
with PSIS before returning shape parameter `ξ`.
"""
function psis_smooth_tail!(tail::AbstractVector{T}, cutoff::T) where {T<:AbstractFloat}
    len = length(tail)
    @turbo @. tail = tail - cutoff

    # save time not sorting since tail is already sorted
    ξ, σ = gpdfit(tail)
    if ξ ≠ Inf
        @turbo @. tail = gpd_quantile(($(1:len) - .5) / len, ξ, σ) + cutoff
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



##########################
#### HELPER FUNCTIONS ####
##########################

"""
Generate the relative effective sample size if not provided by the user.
"""
function _generate_r_eff(weights, dims, r_eff, source)
    if isempty(r_eff)
        if source == "mcmc"
            @info "Adjusting for autocorrelation. If the posterior samples are not " *
            "autocorrelated, specify the source of the posterior sample using the keyword " *
            "argument `source`. MCMC samples are always autocorrelated; VI samples are not."
            return relative_eff(reshape(weights, dims))
        elseif source ∈ SAMPLE_SOURCES
            @info "Samples have not been adjusted for autocorrelation. If the posterior " *
            "samples are autocorrelated, as in MCMC methods, ESS estimates will be " *
            "upward-biased, and standard error estimates will be downward-biased. " *
            "MCMC samples are always autocorrelated; VI samples are not."
            return ones(size(weights)[1])
        else 
            throw(ArgumentError("$source is not a valid source. " *
                "Valid sources are $SAMPLE_SOURCES."
                )
            )
            return ones(size(weights)[1])
        end
    else 
        return r_eff
    end
end

"""
Make sure all inputs to `psis` are valid.
"""
function check_input_validity_psis(
    log_ratios::AbstractArray{T,3},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    if any(isnan, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains NaN values)."))
    elseif any(isinf, log_ratios)
        throw(DomainError("Invalid input for `log_ratios` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `log_ratios` (array is empty)."))
    elseif any(isnan, r_eff)
        throw(ArgumentError("Invalid input for `r_eff` (contains NaN values)."))
    elseif any(isinf, r_eff)
        throw(DomainError("Invalid input for `r_eff` (contains infinite values)."))
    elseif isempty(log_ratios)
        throw(ArgumentError("Invalid input for `r_eff` (array is empty)."))
    elseif length(r_eff) ≠ size(log_ratios, 1)
        throw(ArgumentError("Size of `r_eff` does not equal the number of data points."))
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
                "Unable to fit generalized Pareto distribution: all tail values are the same.
                $LIKELY_ERROR_CAUSES",
            ),
        )
    elseif length(tail) < MIN_TAIL_LEN
        throw(
            ArgumentError(
                "Unable to fit generalized Pareto distribution: tail length was too short.
                $LIKELY_ERROR_CAUSES",
            ),
        )
    end
    return nothing
end


function assume_one_chain(log_ratios)
    @info "Chain information was not provided; " * 
    "all samples are assumed to be drawn from a single chain."
    return ones(length(log_ratios))
end
