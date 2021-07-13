using Base: sign_mask
using ParetoSmooth
using Test
using Statistics
using AxisKeys
using Turing

import RData


let og_array = RData.load("Example_Log_Likelihood_Array.RData")["x"]
    global log_lik_arr = copy(permutedims(og_array, [3, 1, 2]))
end
let og_weights = RData.load("Weight_Matrix.RData")["weightMatrix"]
    global r_weights = exp.(permutedims(reshape(og_weights, 500, 2, 32), [3, 1, 2]))
end
r_eff = RData.load("Rel_Eff.RData")["rel_eff"]
r_psis = RData.load("Psis_Object.RData")["x"]
r_tail_len = Int.(RData.load("Tail_Vector.RData")["tail"])
pareto_k = RData.load("Pareto_K.RData")["pareto_k"]
r_loo = RData.load("Example_Loo.RData")["example_loo"]


# Add labels, reformat
r_loo["pointwise"] = KeyedArray(r_loo["pointwise"][:, Not(4)];
                            data = 1:size(r_loo["pointwise"], 1),
                            statistic=[:est_score, :mcse, :est_overfit, :pareto_k],
                        )

r_loo["estimates"] = KeyedArray(r_loo["estimates"];
                                criterion=[:total_score, :overfit, :avg_score],
                                estimate=[:Estimate, :SE],
                            )
r_loo["estimates"](criterion=:avg_score) .= 
    r_loo["estimates"](criterion=:total_score) / size(r_loo["pointwise"], 1)


@testset "ParetoSmooth.jl" begin

    # All of these should run
    with_r_eff = psis(log_lik_arr, r_eff)
    jul_psis = psis(log_lik_arr)
    log_lik_mat = reshape(log_lik_arr, 32, 1000)
    chain_index = vcat(fill(1, 500), fill(2, 500))
    matrix_psis = psis(log_lik_mat; chain_index=chain_index)
    log_psis = psis(log_lik_arr; log_weights=true)

    jul_loo = loo(log_lik_arr)
    r_eff_loo = psis_loo(log_lik_arr; r_eff=r_eff)
    
    
    # max 10% difference in tail length calc between Julia and R
    @test maximum(abs.(log.(jul_psis.tail_len ./ r_tail_len))) ≤ .1
    @test maximum(abs.(jul_psis.tail_len .- r_tail_len)) ≤ 10
    @test maximum(abs.(with_r_eff.tail_len .- r_tail_len)) ≤ 1
    
    # RMSE from R version is less than .1%
    @test sqrt(mean((with_r_eff.weights ./ r_weights .- 1).^2)) ≤ .001
    # RMSE less than .2% when using InferenceDiagnostics' ESS
    @test sqrt(mean((jul_psis.weights ./ r_weights .- 1).^2)) ≤ .002
    # Max difference is 1%
    @test maximum(log_psis.weights .- log.(r_weights)) ≤ .01


    ## Test difference in loo pointwise results

    # Different r_eff
    errs = (r_loo["pointwise"] - jul_loo.pointwise).^2
    @test sqrt(mean(errs(:est_score))) ≤ .01
    @test sqrt(mean(errs(:est_overfit))) ≤ .01
    @test sqrt(mean(errs(:pareto_k))) ≤ .05
    errs_mcse = log.(r_loo["pointwise"](:mcse) ./ jul_loo.pointwise(:mcse)).^2
    # @test sqrt(mean(errs_mcse)) ≤ .1

    # Same r_eff
    errs = (r_loo["pointwise"] - r_eff_loo.pointwise).^2
    @test sqrt(mean(errs(:est_score))) ≤ .01
    @test sqrt(mean(errs(:est_overfit))) ≤ .01
    @test sqrt(mean(errs(:pareto_k))) ≤ .05
    errs_mcse = log.(r_loo["pointwise"](:mcse) ./ r_eff_loo.pointwise(:mcse)).^2
    # @test sqrt(mean(errs_mcse)) ≤ .1
    
    # Test estimates
    errs = r_loo["estimates"] - jul_loo.estimates
    @test maximum(abs.(errs)) ≤ .01
    
    errs = r_loo["estimates"] - r_eff_loo.estimates
    @test maximum(abs.(errs)) ≤ .01

    # Test for calling correct method
    @test jul_loo.psis_object.weights ≈ psis(-log_lik_arr).weights
    @test r_eff_loo.psis_object.weights ≈ psis(-log_lik_arr, r_eff).weights
end

@testset "compute loo" begin
    using ParetoSmooth, MCMCChains, Distributions, Random
    using Turing

    Random.seed!(112)
    # simulated samples for μ
    samples = randn(50, 1, 3)

    data = randn(50)

    chain = Chains(samples)

    function compute_loglike(μ, data)
        return logpdf(Normal(μ, 1), data)
    end

    loo1 = compute_loo(chain, data, compute_loglike)
    # pass if yields a value
    @test isa(loo1, Float64)

    pw_lls = pointwise_loglikes(samples, data, compute_loglike)
    loo2 = compute_loo(pw_lls)
    @test loo1 ≈ loo2 atol = 1e-6

    @model function model(y)
        μ ~ Normal(0, 1)
        σ ~ truncated(Cauchy(0, 1), 0, Inf)
        y .~ Normal(μ, σ)
    end

    chain = sample(model(data), NUTS(1000, .65), MCMCThreads(), 1000, 4)

    loo = compute_loo(chain, model(data))
    # pass if yields a value
    @test isa(loo, Float64)
end

@testset "pointwise log likelihoods" begin
    using ParetoSmooth, MCMCChains, Distributions, Random
    Random.seed!(112)
    # simulated samples for μ
    samples = randn(1000, 1, 3)

    data = randn(50)

    chain = Chains(samples)

    function compute_loglike(μ, data)
        return logpdf(Normal(μ, 1), data)
    end

    pll1 = pointwise_loglikes(chain, data, compute_loglike)
    pll2 = pointwise_loglikes(samples, data, compute_loglike)

    @test pll1 ≈ pll2 atol = 1e-6
    # data points, samples, chains
    @test size(pll1) == (50, 1000, 3)

    # simulated samples for μ
    samples = randn(1, 1, 1)

    data = randn(50)

    pll3 = pointwise_loglikes(samples, data, compute_loglike)

    @test sum(logpdf.(Normal(samples[1], 1), data)) ≈ sum(pll3) atol = 1e6
end