module ESS

using Base: AbstractFloat
using MCMCChains
using LoopVectorization
using Tullio

"""
    relative_eff(sample::AbstractArray, cores = Threads.nthreads())

Compute the MCMC effective sample size divided by the actual sample size.
"""
function relative_eff(sample::AbstractArray{T,3}) where {T<:AbstractFloat}
    dimensions = size(sample)
    posteriorSampleSize = dimensions[1] * dimensions[2]
    ess, = MCMCChains.ess_rhat(permutedims(sample, [2, 1, 3]))  # Only need ESS, not rhat
    rEff = ess / posteriorSampleSize
    return rEff
end


function psis_n_eff(
    weights::AbstractVector{T},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    @tullio sumSquares := weights[x]^2
    return r_eff / sumSquares
end

function psis_n_eff(
    weights::AbstractMatrix{T},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    @tullio sumSquares[x] := weights[x, y]^2
    return @tturbo r_eff ./ sumSquares
end

function psis_n_eff(
    weights::AbstractArray{T,3},
    r_eff::AbstractVector{T},
) where {T<:AbstractFloat}
    @tullio sumSquares[x] := weights[x, y, z]^2
    return @tturbo r_eff ./ sumSquares
end

function psis_n_eff(weights::AbstractVecOrMat{T}) where {T<:AbstractFloat}
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
    "will be overoptimistic if samples are autocorrelated."
    return psis_n_eff(weights, ones(size(weights)))
end

end
