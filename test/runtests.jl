using ParetoSmooth
using Test
using Statistics

import RData

let og_array = RData.load("Example_Log_Likelihood_Array.RData")["x"]
    global log_lik_arr = copy(permutedims(og_array, [3, 1, 2]))
end
let og_weights = RData.load("Weight_Matrix.RData")["weightMatrix"]
    global r_weights = exp.(permutedims(reshape(og_weights, 500, 2, 32), [3, 1, 2]))
end
rel_eff = RData.load("Rel_Eff.RData")["rel_eff"]
r_psis = RData.load("Psis_Object.RData")["x"]
r_tail_len = Int.(RData.load("Tail_Vector.RData")["tail"])
pareto_k = RData.load("Pareto_K.RData")["pareto_k"]
r_loo = RData.load("Example_Loo.RData")["example_loo"]
r_pointwise = RData.load("Pointwise_Loo.RData")["pointwise"]

@testset "ParetoSmooth.jl" begin

    # All of these should run
    with_rel_eff = psis(log_lik_arr, rel_eff)
    jul_psis = psis(log_lik_arr)
    log_lik_mat = reshape(log_lik_arr, 32, 1000)
    chain_index = vcat(fill(1, 500), fill(2, 500))
    matrixPsis = psis(log_lik_mat; chain_index=chain_index)
    log_psis = psis(log_lik_arr; lw=true)

    jul_loo = loo(log_lik_arr)
    rel_eff_loo = loo(log_lik_arr; rel_eff=rel_eff)

    # At most 1 value is off from R value by more than 1%
    @test count(.!isapprox.(pareto_k, with_rel_eff.pareto_k)) ≤ 1
    
    # max 10% difference in tail length calc between Julia and R
    @test maximum(abs.(log.(jul_psis.tail_len ./ r_tail_len))) ≤ .1
    @test maximum(abs.(jul_psis.tail_len .- r_tail_len)) ≤ 10
    @test maximum(abs.(with_rel_eff.tail_len .- r_tail_len)) ≤ 1
    
    # RMSE from R version is less than .1%
    @test sqrt(mean((with_rel_eff.weights ./ r_weights .- 1).^2)) ≤ .001
    # RMSE less than .2% when using InferenceDiagnostics' ESS
    @test sqrt(mean((jul_psis.weights ./ r_weights .- 1).^2)) ≤ .002
    @test count(with_rel_eff.weights .≉ r_weights) ≤ 10
    @test count(jul_psis.weights .≉ matrixPsis.weights) ≤ 10
    # Max difference is 1%
    @test maximum(log_psis.weights .- log.(r_weights)) ≤ .01

    # Max difference in loo results test
    # @test maximum(abs(jul_loo.estimates .- r_loo["estimates"])) ≤ .05
    # @test maximum(abs(rel_eff_loo.estimates .- r_loo["estimates"])) ≤ .01
    # @test maximum(abs((jul_loo.pointwise) .- r_loo["pointwise"])) ≤ .05
    # @test maximum(abs((rel_eff.pointwise) .- r_loo["pointwise"])) ≤ .01

    # Test for calling correct method

    @test jul_loo.psis_object.weights == psis(-log_lik_arr).weights
    @test rel_eff_loo.psis_object.weights == psis(-log_lik_arr, rel_eff).weights
end
