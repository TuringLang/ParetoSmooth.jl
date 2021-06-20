module PSIS

using LoopVectorization, Base.Threads

export do_psis_i, do_psis_i!

include("GPD.jl")

const MIN_LEN = 5  # Minimum size of a sample for PSIS to work


"""
    do_psis_i!(logRatios::AbstractVector{T <: AbstractFloat}, tailLength::Integer) -> T
Do PSIS on a single vector, smoothing a vector of log weights *in place* before returning the GPD shape parameter `ξ`.


# Arguments
- `logRatios::AbstractVector{T}`: The abstract vector of log importance-sampling ratios to be smoothed.
- `tailLength::Integer`: 


# Returns
- `T <: AbstractFloat`: ξ, the shape parameter for the GPD; larger numbers indicate thicker tails.

# Extended help
Additional information can be found in the LOO package from R {Add link here}
{Add citation for Vehtari paper}
"""
function do_psis_i!(logRatios::AbstractVector{T}, 
                    tailLength::Integer = def_tail_length(logRatios)
                   ) where T<:AbstractFloat
    len = length(logRatios)
    maxRatio = maximum(logRatios)
    # Shift log ratios for safer exponentation
    @turbo @. logRatios = logRatios - maxRatio
    @views sortedRatios = logRatios[sortperm(logRatios)]

    ξ::T = Inf

    if tailLength ≥ MIN_LEN
        # TODO: Faster sorting by presorting all vectors together
        tailStartsAt = len - tailLength + 1  # index of smallest tail value
        @views tail = sortedRatios[tailStartsAt:len]

        if maximum(tail) ≈ minimum(tail)
            @warn("Unable to fit generalized Pareto distribution: all tail values are the same. " * 
            "Falling back on truncated importance sampling.")
        else
            cutoff = sortedRatios[tailStartsAt - 1]  # largest value smaller than tail values
            ξ = psis_smooth_tail!(tail, cutoff)
        end
    else
        @warn("Unable to fit generalized Pareto distribution: Tail length was too short. " * 
        "Falling back on regular importance sampling.")
    end
    # truncate at max of raw wts (i.e., 0 since max has been subtracted)
    @turbo @. logRatios = min(logRatios, 0)
    # shift log weights back so that the smallest log weights remain unchanged
    @turbo @. logRatios = logRatios + maxRatio
    
    return ξ
end


"""
    do_psis_i(logRatios::AbstractVector, tailLength::Real)
Do PSIS on a single vector, returning a named tuple with smoothed log weights and the GPD shape parameter ξ.

The log weights (or log ratios if no smoothing) larger than the largest raw ratio are set to the largest raw ratio.
"""
function do_psis_i(logRatios::AbstractVector{T}, 
                   tailLength::Integer = def_tail_length(logRatios)
                  ) where T <: AbstractFloat
    x = zeros(eltype(logRatios), length(logRatios))
    @turbo x .= logRatios
    ξ = do_psis_i!(x, tailLength)
    return (logRatios = x, k = ξ)
end


"""
    tail_length(logRatios::AbstractVector)
Define the tail length as in Vehtari et al. (2019).
"""
@inline function def_tail_length(logRatios::AbstractVector{T}) where T <: AbstractFloat
    if length(logRatios) > 225
        return 3 * isqrt(length(logRatios))
    else
        return length(logRatios) ÷ 5
    end
end



"""
    psis_smooth_tail(tail::AbstractVector)
Takes an *already sorted* vector of observations from the tail and smooths it in place with PSIS before returning shape parameter `ξ`.
"""
function psis_smooth_tail!(tail::AbstractVector{T}, cutoff::Real) where T <: AbstractFloat
    len = length(tail)
    expCutoff = exp(cutoff)

    # save time not sorting since x already sorted
    ξ, σ = @turbo GPD.gpdfit(exp.(tail) .- expCutoff)
    if ξ != Inf
        @turbo @. tail = log(GPD.gpd_quantile($(1:len) / (len+1), ξ, σ) + expCutoff)
    end
    return ξ
end

end