module LooUtility

const IMPLEMENTED_IS_METHODS = ["psis"]
const MIN_TAIL_LEN = 5  # Minimum size of a tail for PSIS to work
const LIKELY_ERROR_CAUSES = """
1. Bugs in the program that generated the sample, or incorrect input variables.

2. Your chains failed to converge. Check your diagnostics.

3. You do not have enough posterior samples (Less than ~100 samples) -- try sampling more values.
"""

export check_input_validity_psis, IMPLEMENTED_IS_METHODS


######################
# INTERNAL FUNCTIONS #
######################

function check_input_validity_psis(log_ratios::AbstractArray{T, 3}, r_eff::AbstractVector{T}) where {T <: AbstractFloat}
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
    elseif length(r_eff) ≠ size(log_ratios, 2)
        throw(ArgumentError("Invalid input -- size of `r_eff` does not equal the number of chains."))
    end
    return nothing
end

"""
Check the tail to make sure a GPD fit is possible.
"""
function check_tail(tail::AbstractVector{T}) where {T<:AbstractFloat}
    if maximum(tail) ≈ minimum(tail)
        throw(ArgumentError("Unable to fit generalized Pareto distribution: all tail values are the same. \n$LIKELY_ERROR_CAUSES"))
    elseif length(tail) < MIN_TAIL_LEN
        throw(ArgumentError("Unable to fit generalized Pareto distribution: tail length was too short.\n$LIKELY_ERROR_CAUSES"))
    end
    return nothing    
end


end