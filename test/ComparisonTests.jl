using CSV
using DataFrames
using StatsBase

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
            @. result += x[i] * x[i + 1]
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

    chn5_1t = sample(m5_1t(df.A, df.D), NUTS(1000, 0.9), MCMCThreads(), 1000, 4)

    @model function m5_2t(M, D)
        a ~ Normal(0, 0.2)
        bM ~ Normal(0, 0.5)
        σ ~ Exponential(1)
        for i in eachindex(D)
            μ = lin(a, M[i], bM)
            D[i] ~ Normal(μ, σ)
        end
    end

    chn5_2t = sample(m5_2t(df.M, df.D), NUTS(1000, 0.9), MCMCThreads(), 1000, 4)

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

    chn5_3t = sample(m5_3t(df.A, df.M, df.D), NUTS(1000, 0.9), MCMCThreads(), 1000, 4)
    pw_lls5_1t = pointwise_log_likelihoods(m5_1t(df.A, df.D), chn5_1t)
    psis_loo_output5_1t = psis_loo(m5_1t(df.A, df.D), chn5_1t)
    psis_loo_output5_1t |> display

    pw_lls5_2t = pointwise_log_likelihoods(m5_2t(df.M, df.D), chn5_2t)
    psis_loo_output5_2t = psis_loo(m5_2t(df.M, df.D), chn5_2t)
    psis_loo_output5_2t |> display

    pw_lls5_3t = pointwise_log_likelihoods(m5_3t(df.A, df.M, df.D), chn5_3t)
    psis_loo_output5_3t = psis_loo(m5_3t(df.A, df.M, df.D), chn5_3t)
    psis_loo_output5_3t |> display
    
    n_tuple = (
        m5_1t=psis_loo_output5_1t, 
        m5_2t=psis_loo_output5_2t,
        m5_3t=psis_loo_output5_3t,
    )
    comps = loo_compare(n_tuple)
    comps |> display

    @test comps.estimates(:m5_1t, :cv_est) ≈ 0.00 atol = .01
    @test comps.estimates(:m5_1t, :se_cv_est) ≈ 0.00 atol = .01
    @test comps.estimates(:m5_1t, :weight) ≈ 0.67 atol = .01
    @test comps.estimates(:m5_2t, :cv_est) ≈ -6.68 atol = .01
    @test comps.estimates(:m5_2t, :se_cv_est) ≈ 4.79 atol = .01
    @test comps.estimates(:m5_2t, :weight) ≈ 0.00 atol = .01
    @test comps.estimates(:m5_3t, :cv_est) ≈ -0.69 atol = .01
    @test comps.estimates(:m5_3t, :se_cv_est) ≈ 0.42 atol = .01
    @test comps.estimates(:m5_3t, :weight) ≈ 0.33 atol = .01
    @test sum(comps.estimates(:, :weight, :)) ≈ 1
    total = NamedDims.unname(sum(comps.pointwise(:, :cv_est, :); dims=:data))
    @test reshape(total, 3) ≈ comps.estimates(:, :cv_est) atol=.001
    

end
