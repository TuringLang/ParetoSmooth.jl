using AxisKeys
using MCMCChains
using ParetoSmooth
using Statistics
using Test
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
    @testset "Basic Arrays" begin

        # All of these should run
        with_r_eff = psis(log_lik_arr, r_eff)
        jul_psis = psis(log_lik_arr)
        log_lik_mat = reshape(log_lik_arr, 32, 1000)
        chain_index = vcat(fill(1, 500), fill(2, 500))
        matrix_psis = psis(log_lik_mat; chain_index=chain_index)
        log_psis = psis(log_lik_arr; log_weights=true)

        jul_loo = loo(log_lik_arr)
        r_eff_loo = psis_loo(log_lik_arr, r_eff)

        display(jul_loo)
        
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


    @testset "MCMCChains and Turing utilities" begin
        using Distributions, Random
        Random.seed!(112)
        # simulated samples for μ
        samples = randn(100, 1, 1)
        data = randn(50)
        chain = Chains(samples)

        compute_loglike(μ, data) = logpdf(Normal(μ, 1), data)
        compute_loglike(μ, σ, data) = logpdf(Normal(μ, σ), data)
        
        pll1 = pointwise_log_likelihoods(compute_loglike, chain, data)
        pll2 = pointwise_log_likelihoods(compute_loglike, samples, data)
        # the pointwise log likehoods should be the same for both methods
        @test pll1 ≈ pll2 atol = 1e-6
        # test the dimensions: data points, samples, chains
        @test size(pll1) == (50, 100, 1)
        # test that sum of pointwise log likelihoods equals sum of log likelihoods
        @test sum(sum(map(s->logpdf.(Normal(s, 1), data), samples))) ≈ sum(pll1) atol = 1e6
        # test that psis_loo works with MCMCChains and yields correct type
        psis_loo_output = psis_loo(compute_loglike, chain, data)
        @test isa(psis_loo_output, PsisLoo)
        # test that loo works with MCMCChains and yields correct type
        psis_output = loo(compute_loglike, chain, data)
        @test isa(psis_output, PsisLoo)
        # test that psis works with MCMCChains and yields correct type
        psis_output = psis(compute_loglike, chain, data)
        @test isa(psis_output, Psis)

        # ensure that methods work with r_eff argument
        r_eff = similar(pll2, 0)
        # test that psis_loo works with MCMCChains and yields correct type
        psis_loo_output = psis_loo(compute_loglike, chain, data, r_eff)
        @test isa(psis_loo_output, PsisLoo)
        # test that loo works with MCMCChains and yields correct type
        psis_output = loo(compute_loglike, chain, data, r_eff)
        @test isa(psis_output, PsisLoo)
        # test that psis works with MCMCChains and yields correct type
        psis_output = psis(compute_loglike, chain, data, r_eff)
        @test isa(psis_output, Psis)

        @model function model(data)
            μ ~ Normal(0, 1)
            σ ~ truncated(Cauchy(0, 1), 0, Inf)
            for i in eachindex(data)
                data[i] ~ Normal(μ, σ)
            end
        end

        chain = sample(model(data), NUTS(1000, .9), MCMCThreads(), 1000, 4)
        pw_lls_turing = pointwise_log_likelihoods(model(data), chain)
        pw_lls_loglike = pointwise_log_likelihoods(compute_loglike, chain, data)

        # test the dimensions: data points, samples, chains
        @test size(pw_lls_turing) == (50, 1000, 4)
        # test that sum of pointwise log likelihoods equals sum of log likelihoods
        @test sum(sum(map(s->logpdf.(Normal(s, 1), data), samples))) ≈ sum(pw_lls_turing) atol = 1e6
        # Turing would work the same as compute_loglike
        @test pw_lls_loglike ≈ pw_lls_turing atol = 1e6

        # test that psis_loo works with Turing model and MCMCChains and yields correct type
        psis_loo_output = psis_loo(model(data), chain)
        @test isa(psis_loo_output, PsisLoo)
        # test that loo works with Turing model and MCMCChains and yields correct type
        psis_output = loo(model(data), chain)
        @test isa(psis_output, PsisLoo)
        # test that psis works with Turing model and MCMCChains and yields correct type
        psis_output = psis(model(data), chain)
        @test isa(psis_output, Psis)


        
        # ensure that methods work with r_eff argument
        r_eff = similar(pw_lls_turing, 0)
        # test that psis_loo works with Turin and gives correct type
        psis_loo_output = psis_loo(model(data), chain, r_eff)
        @test isa(psis_loo_output, PsisLoo)
        # test that loo works with Turing model and MCMCChains and gives correct type
        psis_output = loo(model(data), chain, r_eff)
        @test isa(psis_output, PsisLoo)
        # test that psis works with Turing model and MCMCChains and gives correct type
        psis_output = psis(model(data), chain, r_eff)
        @test isa(psis_output, Psis)

    end
end
