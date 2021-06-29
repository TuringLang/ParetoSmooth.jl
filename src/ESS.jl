module ESS

using Base: AbstractFloat
using MCMCChains
using LoopVectorization
using Tullio

export relative_eff, psis_n_eff

"""
    relative_eff(sample::AbstractArray{AbstractFloat, 3})

Compute the MCMC effective sample size divided by the nominal sample size.
"""
function relative_eff(sample::AbstractArray{T,3}) where {T<:AbstractFloat}
    dimensions = size(sample)
    posteriorSampleSize = dimensions[2] * dimensions[3]
    # Only need ESS, not rhat
    ess, = MCMCChains.ess_rhat(permutedims(sample, [2, 1, 3]))  
    rEff = ess / posteriorSampleSize
    return rEff
end

"""
    function psis_n_eff(
        weights::AbstractVector{T},
        r_eff::AbstractVector{T}
    ) -> AbstractVector{T}
"""
function psis_n_eff(
    weights::AbstractVector{T},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    @tullio sumSquares := weights[x]^2
    return r_eff ./ sumSquares
end

function psis_n_eff(
    weights::AbstractMatrix{T},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    @tullio sumSquares[x] := weights[x, y]^2
    return @tturbo r_eff ./ sumSquares
end

function psis_n_eff(weights::AbstractArray{T}) where {T<:AbstractFloat}
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
    "will be overoptimistic if samples are autocorrelated."
    return psis_n_eff(weights, ones(size(weights)))
end

end
