using AxisKeys
using LoopVectorization
using StatsBase
using Tables
using Tullio

"""
    adapt_moments(
        log_target::Function,
        psis_object::Psis,
        samples::AbstractArray,
        data;
        hard_thresh::Real = 2/3,
        soft_thresh::Real = 1/2,
        soft_cap::Integer = 10
    )

Perform importance-weighted moment matching, adapting a sample from a proposal distribution
to more closely match the target distribution.

# Arguments
  - `log_target`: The log-pdf of the target distribution, described as a function of the 
"""
function adapt_moments(
    log_target::Function,
    psis_object::Psis,
    samples::AbstractArray,
    data;
    hard_thresh::Real = 2/3,
    soft_thresh::Real = 1/2,
    soft_cap::Integer = 10
)

    psis_object = Psis()
    dims = size(samples)
    n_steps, n_params, n_chains = dims
    mcmc_count = n_steps * n_chains
    weights = psis_object.weights
    ξ = psis_object.pareto_k
    resample_count = size(weights, 1)
    
    Threads.@threads @inbounds for resample in 1:resample_count
        @views log_proposal = weights[resample, :, :]
        _match!(log_target, log_proposal, samples, ξ; hard_thresh, soft_thresh, soft_cap)
    end

end


function _match!(
    log_target::Function,
    log_proposal::AbstractArray,
    θ_hats::AbstractArray,
    ξ::Real,
    hard_thresh::Real = 2/3,
    soft_thresh::Real = 1/2,
    soft_cap::Integer = 10,
)
    dims = size(θ_hats)
    mcmc_count = size(θ_hats, :parameter)

    # initialize variables
    num_iter = 0  # iterations of IWMM
    transform = 1
    θ_proposed = similar(θ_hats)
    ξ_proposed = soft_thresh
    μ = mean(θ_hats; dims=:parameter)
    μ_proposed = similar(μ)
    σ = std(θ_hats; dims=:parameter)
    σ_proposed = similar(σ)
    weights = 
    weights_proposed = similar(log_proposal)
    

    while _keep_going(ξ, hard_thresh, soft_thresh, soft_cap, num_iter)
        
        
        if transform == 1
            μ_proposed = mean(θ_hats, weights; dims=2)
            σ_proposed .= σ
        elseif transform == 2
            μ_proposed = mean(θ_hats, weights; dims=2)
            σ_proposed = std(θ_hats, weights; mean=μ_proposed, dims=2)
        elseif transform == 3
            μ_proposed, Σ_proposed = mean_and_cov(θ_hats, weights; dims=2)
            σ_proposed = sqrt(Σ_proposed)
        elseif transform == 4
            break
        end

        θ_proposed = (θ_hats + μ_proposed - μ) * (σ_proposed * inv(σ)) 
        @. weights_proposed = _safe_exp(log_target(θ_proposed) - log_proposal)
        ξ_proposed = _psis_smooth!(weights_proposed)

        if ξ_proposed < ξ
            num_iter += 1
            _normalize!(weights_proposed)

            ξ = ξ_proposed
            μ = μ_proposed
            σ = σ_proposed
            θ_hats = θ_proposed
        else
            transform += 1
        end

    end

end


function _keep_going(
    ξ::Real, 
    hard_thresh::Real, 
    soft_thresh::Real, 
    soft_cap::Int, 
    num_iter::Integer
)
    if ξ > hard_thresh
        return true
    elseif (ξ > soft_thresh) && (num_iter ≤ soft_cap)
        return true
    else 
        return false
    end
end



"""
Safely exponentiate x, preventing underflow/overflow by rescaling all elements 
by a common factor
"""
function _safe_exp(x)
    return exp(x - $maximum(x; dims=2) + log(length(x)))
end