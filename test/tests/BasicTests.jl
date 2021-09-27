using AxisKeys
using NamedDims
using Statistics
import RData

@testset "Basic Arrays" begin

    let og_array = RData.load("data/Example_Log_Likelihood_Array.RData")["x"]
        global log_lik_arr = copy(permutedims(og_array, [3, 1, 2]))
    end
    let og_weights = RData.load("data/Weight_Matrix.RData")["weightMatrix"]
        global r_weights = exp.(permutedims(reshape(og_weights, 500, 2, 32), [3, 1, 2]))
    end
    r_eff = RData.load("data/Rel_Eff.RData")["rel_eff"]
    r_psis = RData.load("data/Psis_Object.RData")["x"]
    r_tail_len = Int.(RData.load("data/Tail_Vector.RData")["tail"])
    pareto_k = RData.load("data/Pareto_K.RData")["pareto_k"]
    r_loo = RData.load("data/Example_Loo.RData")["example_loo"]
    
    
    # Add labels, reformat
    r_pointwise = KeyedArray(
        r_loo["pointwise"][:, Not(4)];
        data=1:size(r_loo["pointwise"], 1),
        statistic=[:cv_elpd, :mcse, :p_eff, :pareto_k],
    )
    
    r_loo["estimates"] = hcat(r_loo["estimates"], r_loo["estimates"] / size(r_pointwise, 1))
    r_ests = KeyedArray(
        r_loo["estimates"][Not(3), :];
        statistic=[:cv_elpd, :p_eff],
        column=[:total, :se_total, :mean, :se_mean],
    )

    # All of these should run
    with_r_eff = psis(log_lik_arr; r_eff=r_eff)
    jul_psis = psis(log_lik_arr)
    log_lik_mat = reshape(log_lik_arr, 32, 1000)
    chain_index = vcat(fill(1, 500), fill(2, 500))
    matrix_psis = psis(log_lik_mat; chain_index=chain_index)

    jul_loo = psis_loo(log_lik_arr)
    r_eff_loo = psis_loo(log_lik_arr; r_eff=r_eff)

    @test display(jul_psis) === nothing
    @test display(jul_loo) === nothing

    # max 20% difference in tail length calc between Julia and R
    @test maximum(abs.(log.(jul_psis.tail_len ./ r_tail_len))) ≤ 0.2
    @test maximum(abs.(jul_psis.tail_len .- r_tail_len)) ≤ 10
    @test maximum(abs.(with_r_eff.tail_len .- r_tail_len)) ≤ 2

    # RMSE from R version is less than .1%
    @test sqrt(mean((with_r_eff.weights ./ r_weights .- 1) .^ 2)) ≤ 0.001
    # RMSE less than .2% when using InferenceDiagnostics' ESS
    @test sqrt(mean((jul_psis.weights ./ r_weights .- 1) .^ 2)) ≤ 0.002
    # Max difference is 1%
    @test maximum(log.(jul_psis.weights) .- log.(r_weights)) ≤ 0.02


    ## Test difference in loo pointwise results

    # Different r_eff
    jul_pointwise = jul_loo.pointwise([:cv_elpd, :mcse, :p_eff, :pareto_k])
    errs = (r_pointwise - jul_pointwise) .^ 2
    @test sqrt(mean(errs(:cv_elpd))) ≤ 0.01
    @test sqrt(mean(errs(:p_eff))) ≤ 0.01
    @test sqrt(mean(errs(:pareto_k))) ≤ 0.025
    display(r_pointwise(:mcse))
    display(jul_loo.pointwise(:mcse))
    errs_mcse = log.(r_pointwise(:mcse) ./ jul_loo.pointwise(:mcse))
    display(errs_mcse)
    @test sqrt(mean(errs_mcse.^2)) ≤ 0.1

    # Same r_eff
    r_eff_pointwise = r_eff_loo.pointwise([:cv_elpd, :mcse, :p_eff, :pareto_k])
    errs = (r_pointwise - r_eff_pointwise) .^ 2
    @test sqrt(mean(errs(:cv_elpd))) ≤ 0.01
    @test sqrt(mean(errs(:p_eff))) ≤ 0.01
    @test sqrt(mean(errs(:pareto_k))) ≤ 0.025
    errs_mcse = log.(r_pointwise(:mcse) ./ r_eff_loo.pointwise(:mcse))
    display(r_pointwise(:mcse))
    display(r_eff_loo.pointwise(:mcse))
    display(errs_mcse)
    @test sqrt(mean(errs_mcse.^2)) ≤ 0.1

    # Test estimates
    errs = r_ests - jul_loo.estimates(; statistic=[:cv_elpd, :p_eff])
    display(r_ests)
    display(jul_loo.estimates(; statistic=[:cv_elpd, :p_eff]))
    display(errs)
    @test maximum(abs.(errs)) ≤ 0.01

    errs = r_ests - r_eff_loo.estimates(; statistic=[:cv_elpd, :p_eff])
    display(errs)
    @test maximum(abs.(errs)) ≤ 0.01

    # Test for calling correct method
    @test jul_loo.psis_object.weights ≈ psis(-log_lik_arr).weights
    @test r_eff_loo.psis_object.weights ≈ psis(-log_lik_arr; r_eff=r_eff).weights

    @test ParetoSmooth.naive_lpd(log_lik_arr) ≈ jul_loo.estimates(:naive_lpd, :total)
    @test ParetoSmooth.naive_lpd(log_lik_arr) ≈ r_eff_loo.estimates(:naive_lpd, :total)
end