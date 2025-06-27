# Turing Example 

This example demonstrates how to correctly compute PSIS LOO for a model developed with [Turing.jl](https://turinglang.org/stable/). Below, we show two ways to correctly specify the model in Turing. What is most important is to specify the model so that pointwise log densities are computed for each observation. 

To make things simple, we will use a Gaussian model in each example. Suppose observations ``Y = \{y_1,y_2,\dots y_n\}`` come from a Gaussian distribution with an uknown parameter ``\mu`` and known parameter ``\sigma=1``. The model can be stated as follows:

``\mu \sim \mathrm{Normal}(0, 1)``

``Y \sim \mathrm{Normal}(\mu, 1)``

## For Loop Method

One way to specify a model to correctly compute PSIS LOO is to iterate over the observations using a for loop, as follows:
```julia
using Turing
using ParetoSmooth
using Distributions
using Random

Random.seed!(5)

@model function model(data)
    μ ~ Normal()
    for i in 1:length(data)
        data[i] ~ Normal(μ, 1)
    end
end

data = rand(Normal(0, 1), 100)

chain = sample(model(data), NUTS(), 1000)
psis_loo(model(data), chain)
```
The output below correctly indicates PSIS LOO was computed with 100 data points. 

```julia
[ Info: No source provided for samples; variables are assumed to be from a Markov Chain. If the samples are independent, specify this with keyword argument `source=:other`.
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 100 data points. Total Monte Carlo SE of 0.064.
┌───────────┬─────────┬──────────┬───────┬─────────┐
│           │   total │ se_total │  mean │ se_mean │
├───────────┼─────────┼──────────┼───────┼─────────┤
│   cv_elpd │ -158.82 │     9.24 │ -1.59 │    0.09 │
│ naive_lpd │ -157.43 │     9.05 │ -1.57 │    0.09 │
│     p_eff │    1.39 │     0.19 │  0.01 │    0.00 │
└───────────┴─────────┴──────────┴───────┴─────────┘
```
## Dot Vectorization Method

The other method uses dot vectorization in the sampling statement: `.~`. Adapting the model above accordingly, we have:

```julia
using Turing
using ParetoSmooth
using Distributions
using Random

Random.seed!(5)

@model function model(data)
    μ ~ Normal()
    data .~ Normal(μ, 1)
end

data = rand(Normal(0, 1), 100)

chain = sample(model(data), NUTS(), 1000)
psis_loo(model(data), chain)
```
As before, the output correctly indicates PSIS LOO was computed with 100 observations. 
```julia
[ Info: No source provided for samples; variables are assumed to be from a Markov Chain. If the samples are independent, specify this with keyword argument `source=:other`.
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 100 data points. Total Monte Carlo SE of 0.053.
┌───────────┬─────────┬──────────┬───────┬─────────┐
│           │   total │ se_total │  mean │ se_mean │
├───────────┼─────────┼──────────┼───────┼─────────┤
│   cv_elpd │ -158.71 │     9.23 │ -1.59 │    0.09 │
│ naive_lpd │ -157.44 │     9.06 │ -1.57 │    0.09 │
│     p_eff │    1.27 │     0.18 │  0.01 │    0.00 │
└───────────┴─────────┴──────────┴───────┴─────────┘
```

## Incorrect Model Specification

Although the model below is valid, it will not produce the correct results for PSIS LOO because it computes a single log likelihood for the data rather than one for each observation. Note the lack of `.` in the sampling statement.

```julia
using Turing
using ParetoSmooth
using Distributions
using Random

Random.seed!(5)

@model function model(data)
    μ ~ Normal()
    data ~ Normal(μ, 1)
end

data = rand(Normal(0, 1), 100)

chain = sample(model(data), NUTS(), 1000)
psis_loo(model(data), chain)
```

In this case, there is only 1 data point and the standard errors cannot be computed:

```julia 
[ Info: No source provided for samples; variables are assumed to be from a Markov Chain. If the samples are independent, specify this with keyword argument `source=:other`.
┌ Warning: Some Pareto k values are high (>.7), indicating PSIS has failed to approximate the true distribution.
└ @ ParetoSmooth ~/.julia/packages/ParetoSmooth/AJM3j/src/InternalHelpers.jl:46
Results of PSIS-LOO-CV with 1000 Monte Carlo samples and 1 data points. Total Monte Carlo SE of 0.15.
┌───────────┬─────────┬──────────┬─────────┬─────────┐
│           │   total │ se_total │    mean │ se_mean │
├───────────┼─────────┼──────────┼─────────┼─────────┤
│   cv_elpd │ -158.57 │      NaN │ -158.57 │     NaN │
│ naive_lpd │ -157.91 │      NaN │ -157.91 │     NaN │
│     p_eff │    0.66 │      NaN │    0.66 │     NaN │
└───────────┴─────────┴──────────┴─────────┴─────────┘
```