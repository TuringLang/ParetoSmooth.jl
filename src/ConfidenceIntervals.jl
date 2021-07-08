module ConfidenceIntervals
    # You shouldn't be able to see this; if you can, I uploaded it by mistake.
    ev_boot = bbmean(pointwise_ev, boostrap_count)
    eff_p_boot = bbmean(data_size * pointwise_naive, boostrap_count)
    ev_low, ev_high = quantile(bootstrap_dist, ci; alpha=0)
    eff_p_low, eff_p_high = quantile(bootstrap_dist, ci; alpha=0)


    function bbmean(pointwise_ev::T, bootstrap_count::Integer) where {T<:AbstractArray}
        # antithetic bootstrap for variance reduction -- we use 2 * bootstrap_count resamples
        data_size = length(pointwise_ev)
        weights = similar(pointwise_ev, (bootstrap_count*2, data_size))
        weights[1:bootstrap_count] .= rand(Dirichlet(ones(data_size)), bootstrap_count)
        rightweights = (bootstrap_count+1):(2*bootstrap_count)
        weights[rightweights] .= (1 .- weights[1:bootstrap_count]) / (data_size - 1)
        return @tullio bb_mean[i] := weights[i, j] * pointwise_ev[i] |> _ / data_size
    end
    
end