module PSIS

using LoopVectorization, Tullio, GPD

export psis

const MIN_LEN = 5  # Minimum size of a sample for PSIS to work


"""
    do_psis_i(logRatios::AbstractVector, tailLength::Real)
Do PSIS on a single vector, returning a vector of log weights and the GPD shape parameter ξ.

The log weights (or log ratios if no smoothing) larger than the largest raw ratio are set to the largest raw ratio.
"""
function do_psis_i(logRatios::AbstractVector{T>:Real}, tailLength::Real)
    len = length(logRatios)
    # Shift log ratios for safer exponentation
    logRatios = logRatios .- max(logRatios)
    ξ::Real = Inf

    if tailLength ≥ MIN_LEN
        # TODO: Faster sorting by presorting all vectors together
        ordering = sortperm(logRatios)
        smallestTailValue = len - tailLength  # Smallest value in the tail
        index = smallestTailValue:len
        tail = logRatios[ordering[index]]
        if max(tail) ≈ min(tail)  # If max/min < 1+sqrt(ϵ), numerical stability can't be guaranteed.
            @warn("Unable to fit generalized Pareto distribution: all tail values are the same. " * 
            "Falling back on truncated importance sampling.")
        else
            cutoff = logRatios[ordering[smallestTailValue - 1]] # largest value smaller than tail values
            tail, ξ = psis_smooth_tail(tail, cutoff)
            @turbo logRatios[ordering[index]] .= tail
        end
    else
        @warn("Unable to fit generalized Pareto distribution: Tail length was too short. " * 
        "Falling back on regular importance sampling.")
    end
    # truncate at max of raw wts (i.e., 0 since max has been subtracted)
    @turbo logRatios = @. max(logRatios, 0)
    # shift log weights back so that the smallest log weights remain unchanged
    @turbo logRatios .= logRatios .+ max(logRatios)

    return logRatios, ξ
end


"""
    psis_smooth_tail(tail::AbstractVector)
Takes in an *already sorted* vector of observations from the tail, then returns a named tuple with values `tail` and `ξ`.
"""
function psis_smooth_tail(tail::AbstractVector, cutoff::Real)
    len = length(tail)
    expCutoff = exp(cutoff)

    # save time not sorting since x already sorted
    ξ, σ = @turbo @. $gpdfit(exp(tail) - expCutoff)
    if ξ != Inf
        tail = @turbo @. log(gpd_quantile($(1:len) / (len+1), ξ, σ) + expCutoff)
    end
    return tail, ξ
end

end