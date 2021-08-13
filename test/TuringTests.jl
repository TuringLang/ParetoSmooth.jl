using Turing

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
    @test sum(sum(map(s -> logpdf.(Normal(s, 1), data), samples))) ≈ sum(pll1) atol =
        1e-6
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
    psis_loo_output = psis_loo(compute_loglike, chain, data; r_eff=r_eff)
    @test isa(psis_loo_output, PsisLoo)
    # test that loo works with MCMCChains and yields correct type
    psis_output = loo(compute_loglike, chain, data; r_eff=r_eff)
    @test isa(psis_output, PsisLoo)
    # test that psis works with MCMCChains and yields correct type
    psis_output = psis(compute_loglike, chain, data; r_eff=r_eff)
    @test isa(psis_output, Psis)

    @model function model(data)
        μ ~ Normal(0, 1)
        σ ~ truncated(Cauchy(0, 1), 0, Inf)
        for i in eachindex(data)
            data[i] ~ Normal(μ, σ)
        end
    end

    chain = sample(model(data), NUTS(1000, 0.9), MCMCThreads(), 1000, 4)
    pw_lls_turing = pointwise_log_likelihoods(model(data), chain)
    pw_lls_loglike = pointwise_log_likelihoods(compute_loglike, chain, data)

    # test the dimensions: data points, samples, chains
    @test size(pw_lls_turing) == (50, 1000, 4)
    # test that sum of pointwise log likelihoods equals sum of log likelihoods
    turing_samples = Array(Chains(chain, :parameters).value)
    # make this more terse with @tullio or other method later
    LL = 0.0
    n_samples, n_parms, n_chains = size(turing_samples)
    for s in 1:n_samples
        for c in 1:n_chains
            LL += sum(logpdf.(Normal(turing_samples[s, :, c]...), data))
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
    psis_loo_output = psis_loo(model(data), chain; r_eff=r_eff)
    @test isa(psis_loo_output, PsisLoo)
    # test that loo works with Turing model and MCMCChains and gives correct type
    psis_output = loo(model(data), chain; r_eff=r_eff)
    @test isa(psis_output, PsisLoo)
    # test that psis works with Turing model and MCMCChains and gives correct type
    psis_output = psis(model(data), chain; r_eff=r_eff)
    @test isa(psis_output, Psis)

end