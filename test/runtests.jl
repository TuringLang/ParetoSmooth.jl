using AxisKeys
using MCMCChains
using ParetoSmooth
using Statistics
using Test
using CSV
using DataFrames
using StatsBase
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
r_pointwise = KeyedArray(
    r_loo["pointwise"][:, Not(4)];
    data = 1:size(r_loo["pointwise"], 1),
    statistic=[:loo_est, :mcse, :overfit, :pareto_k],
)

r_loo["estimates"] = hcat(r_loo["estimates"], r_loo["estimates"] / size(r_pointwise, 1))
r_ests = KeyedArray(
    r_loo["estimates"][Not(3), :];
    criterion=[:loo_est, :overfit],
    statistic=[:total, :se_total, :mean, :se_mean],
)

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

        @test display(jul_psis) === nothing
        @test display(jul_loo) === nothing
        
        # max 20% difference in tail length calc between Julia and R
        @test maximum(abs.(log.(jul_psis.tail_len ./ r_tail_len))) ≤ .2
        @test maximum(abs.(jul_psis.tail_len .- r_tail_len)) ≤ 10
        @test maximum(abs.(with_r_eff.tail_len .- r_tail_len)) ≤ 2
        
        # RMSE from R version is less than .1%
        @test sqrt(mean((with_r_eff.weights ./ r_weights .- 1).^2)) ≤ .001
        # RMSE less than .2% when using InferenceDiagnostics' ESS
        @test sqrt(mean((jul_psis.weights ./ r_weights .- 1).^2)) ≤ .002
        # Max difference is 1%
        @test maximum(log_psis.weights .- log.(r_weights)) ≤ .01


        ## Test difference in loo pointwise results

        # Different r_eff
        jul_pointwise = jul_loo.pointwise([:loo_est, :mcse, :overfit, :pareto_k])
        errs = (r_pointwise - jul_pointwise).^2
        @test sqrt(mean(errs(:loo_est))) ≤ .01
        @test sqrt(mean(errs(:overfit))) ≤ .01
        @test sqrt(mean(errs(:pareto_k))) ≤ .025
        errs_mcse = log.(r_pointwise(:mcse) ./ jul_loo.pointwise(:mcse)).^2
        @test_broken sqrt(mean(errs_mcse)) ≤ .1

        # Same r_eff
        r_eff_pointwise = r_eff_loo.pointwise([:loo_est, :mcse, :overfit, :pareto_k])
        errs = (r_pointwise - r_eff_pointwise).^2
        @test sqrt(mean(errs(:loo_est))) ≤ .01
        @test sqrt(mean(errs(:overfit))) ≤ .01
        @test sqrt(mean(errs(:pareto_k))) ≤ .025
        errs_mcse = log.(r_pointwise(:mcse) ./ r_eff_loo.pointwise(:mcse)).^2
        @test_broken sqrt(mean(errs_mcse)) ≤ .1
        
        # Test estimates
        errs = r_ests - jul_loo.estimates(criterion=[:loo_est, :overfit])
        display(r_ests)
        display(jul_loo.estimates(criterion=[:loo_est, :overfit]))
        display(errs)
        @test maximum(abs.(errs)) ≤ .01
        
        errs = r_ests - r_eff_loo.estimates(criterion=[:loo_est, :overfit])
        display(errs)
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
        @test sum(sum(map(s->logpdf.(Normal(s, 1), data), samples))) ≈ sum(pll1) atol = 1e-6
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
        turing_samples = Array(Chains(chain, :parameters).value)
        # make this more terse with @tullio or other method later
        LL = 0.0
        n_samples,n_parms,n_chains = size(turing_samples)
        for s in 1:n_samples
            for c in 1:n_chains
                LL += sum(logpdf.(Normal(turing_samples[s,:,c]...), data))
            end
        end
        @test LL ≈ sum(pw_lls_turing) atol = 1e-6
        # Turing should work the same as compute_loglike
        @test pw_lls_loglike ≈ pw_lls_turing atol = 1e-6

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

    @testset "LooCompare" begin
        using Random
        Random.seed!(129111)

        df = CSV.read(joinpath("data", "WaffleDivorce.csv"), DataFrame)
        df.D = zscore(df.Divorce)
        df.M = zscore(df.Marriage)
        df.A = zscore(df.MedianAgeMarriage)
        data = (D=df.D, A=df.A)

        function lin(a, b, c, x...)
            result = @. a + b * c
            for i in 1:2:length(x)
                @. result += x[i] * x[i+1]
            end
            return result
        end


        @model function m5_1t(A, D)
            a ~ Normal(0, 0.2)
            bA ~ Normal(0, 0.5)
            σ ~ Exponential(1)
            for i in eachindex(D)
                μ = lin(a, A[i], bA)
                D[i] ~ Normal(μ, σ)
            end
        end

        chn5_1t = sample(m5_1t(df.A, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)

        @model function m5_2t(M, D)
            a ~ Normal(0, 0.2)
            bM ~ Normal(0, 0.5)
            σ ~ Exponential(1)
            for i in eachindex(D)
                μ = lin(a, M[i], bM)
                D[i] ~ Normal(μ, σ)
            end
        end

        chn5_2t = sample(m5_2t(df.M, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)

        @model function m5_3t(A, M, D)
            a ~ Normal(0, 0.2)
            bA ~ Normal(0, 0.5)
            bM ~ Normal(0, 0.5)
            σ ~ Exponential(1)
            for i in eachindex(D)
                μ = a + M[i] * bM + A[i] * bA
                D[i] ~ Normal(μ, σ)
            end
        end

        chn5_3t = sample(m5_3t(df.A, df.M, df.D), NUTS(1000, .9), MCMCThreads(), 1000, 4)
        pw_lls5_1t = pointwise_log_likelihoods(m5_1t(df.A, df.D), chn5_1t)
        psis_loo_output5_1t = psis_loo(m5_1t(df.A, df.D), chn5_1t)
        psis_loo_output5_1t |> display

        pw_lls5_2t = pointwise_log_likelihoods(m5_2t(df.M, df.D), chn5_2t)
        psis_loo_output5_2t = psis_loo(m5_2t(df.M, df.D), chn5_2t)
        psis_loo_output5_2t |> display

        pw_lls5_3t = pointwise_log_likelihoods(m5_3t(df.A, df.M, df.D), chn5_3t)
        psis_loo_output5_3t = psis_loo(m5_3t(df.A, df.M, df.D), chn5_3t)
        psis_loo_output5_3t |> display

        loo = loo_compare([pw_lls5_1t, pw_lls5_2t, pw_lls5_3t];
            model_names=[:m5_1t, :m5_2t, :m5_3t])
        loo |> display

        @test loo.table(:m5_1t, :loo_score_diff) ≈ 0.00 atol = 0.05
        @test loo.table(:m5_1t, :se_loo_score_diff) ≈ 0.00 atol = 0.05
        @test loo.table(:m5_1t, :weight) ≈ 0.67 atol = 0.05
        @test loo.table(:m5_2t, :loo_score_diff) ≈ -6.68 atol = 0.05
        @test loo.table(:m5_2t, :se_loo_score_diff) ≈ 4.74 atol = 0.05
        @test loo.table(:m5_2t, :weight) ≈ 0.00 atol = 0.05
        @test loo.table(:m5_3t, :loo_score_diff) ≈ -0.69 atol = 0.05
        @test loo.table(:m5_3t, :se_loo_score_diff) ≈ 0.42 atol = 0.05
        @test loo.table(:m5_3t, :weight) ≈ 0.33 atol = 0.05

        nt = (m5_1t=loo.psis[1], m5_3t=loo.psis[2], m5_2t=loo.psis[3])
        loo2 = loo_compare(nt)
        loo2 |> display

        @test loo2.table(:m5_1t, :loo_score_diff) ≈ 0.00 atol = 0.05
        @test loo2.table(:m5_1t, :se_loo_score_diff) ≈ 0.00 atol = 0.05
        @test loo2.table(:m5_1t, :weight) ≈ 0.67 atol = 0.05
        @test loo2.table(:m5_2t, :loo_score_diff) ≈ -6.68 atol = 0.05
        @test loo2.table(:m5_2t, :se_loo_score_diff) ≈ 4.74 atol = 0.05
        @test loo2.table(:m5_2t, :weight) ≈ 0.00 atol = 0.05
        @test loo2.table(:m5_3t, :loo_score_diff) ≈ -0.69 atol = 0.05
        @test loo2.table(:m5_3t, :se_loo_score_diff) ≈ 0.42 atol = 0.05
        @test loo2.table(:m5_3t, :weight) ≈ 0.33 atol = 0.05

    end
    
end
