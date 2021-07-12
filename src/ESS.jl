using FFTW
using MCMCChains
using LoopVectorization
using Tullio

export relative_eff, psis_n_eff

"""
    relative_eff(sample::AbstractArray{AbstractFloat, 3}; method=FFTESSMethod())

Compute the MCMC effective sample size divided by the nominal sample size.
"""
function relative_eff(
    sample::AbstractArray{T,3}; method=FFTESSMethod()
) where {T<:AbstractFloat}
    dims = size(sample)
    post_sample_size = dims[2] * dims[3]
    # Only need ESS, not rhat
    ess_sample = inv.(permutedims(sample, [2, 1, 3]))
    ess, = MCMCChains.ess_rhat(ess_sample; method=method, maxlag=dims[2])
    r_eff = ess / post_sample_size
    return r_eff
end

"""
    function psis_n_eff(
        weights::AbstractVector{T},
        r_eff::AbstractVector{T}
    ) -> AbstractVector{T}
"""
function psis_n_eff(
    weights::AbstractVector{T}, r_eff::AbstractVector{T}
) where {T<:AbstractFloat}
    @tullio sum_of_squares := weights[x]^2
    return r_eff ./ sum_of_squares
end

function psis_n_eff(
    weights::AbstractMatrix{T}, r_eff::AbstractVector{T}
) where {T<:AbstractFloat}
    @tullio sum_of_squares[x] := weights[x, y]^2
    return @tturbo r_eff ./ sum_of_squares
end

function psis_n_eff(weights::AbstractArray{T}) where {T<:AbstractFloat}
    @warn "PSIS ESS not adjusted based on MCMC ESS. MCSE and ESS estimates " *
          "will be overoptimistic if samples are autocorrelated."
    return psis_n_eff(weights, ones(size(weights)))
end
