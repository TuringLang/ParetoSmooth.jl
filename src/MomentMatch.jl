using AxisKeys
using LoopVectorization
using Tullio


function adapt_weights!(
    log_target::Function,
    psis_object::Psis,
    samples::AbstractArray,
    data
)

    dims = size(samples)
    n_steps, n_params, n_chains = dims
    mcmc_count = n_steps * n_chains
    weights = psis_object.weights
    resample_count = size(weights, 1)
    
    Threads.@threads for resample in 1:resample_count
        @views log_proposal = weights[resample, :, :]
    end

end


function _moment_match_i!(
    log_target::Function,
    log_proposal::AbstractArray,
    θ_hats::AbstractVector, # parameter vector
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
    

    while _keep_going(ξ, hard_thresh, soft_thresh, soft_cap, num_iter)
        
        μ_proposed = _calc_loc(weights, θ_hats, mcmc_count)
        if transform == 1
            σ_proposed .= σ
        elseif transform == 2
            σ_proposed = std(θ_hats)
        elseif transform == 3
            σ_proposed = _calc_scatter(weights, θ_hats, mcmc_count)
        elseif transform == 4
            break
        end

        θ_proposed = (θ_hats + μ_proposed - μ) * (σ_proposed * inv(σ))
        log_like_proposed = log_target(θ_proposed) - log_proposal
        @. weights_proposed = _safe_exp(log_like_proposed)
        ξ_proposed = _psis_smooth!(weights_proposed)

        if ξ_proposed < ξ
            num_iter += 1
            _normalize!(weights_proposed)

            ξ = ξ_proposed
            μ = μ_proposed
            σ = σ_proposed
            θ_hats = θ_proposed
            log_likelihood = log_like_proposed
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
Safely exponentiate -- subtract maximum to prevent overflow
"""
function _safe_exp(x)
    return exp(x - $maximum(x; dims=2))
end