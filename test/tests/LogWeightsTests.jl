using Test

@testset "Log Weights Handling (simplified)" begin
    # Generate random positive weights and compare psis on weights vs log-weights
    using Random
    Random.seed!(1234)
    n = 200
    raw_weights = randexp(n) .+ 1e-3  # positive weights

    # Scale to max 1 as expected by psis
    weights = raw_weights ./ maximum(raw_weights)

    # Copy and run psis on raw weights
    w_copy = copy(weights)
    ξ_w = psis!(w_copy; log_weights=false)

    # Run psis on logs (centered so the maximum log is 0)
    log_weights = log.(weights)
    lw_copy = copy(log_weights)
    ξ_lw = psis!(lw_copy; log_weights=true)

    # After processing, w_copy is in ratio-space (max 1), lw_copy should be in log-space (max 0)
    # Convert lw_copy back to ratio space for comparison
    # Find its maximum (should be ~0) and exponentiate
    max_lw = maximum(lw_copy)
    restored_from_log = exp.(lw_copy .- max_lw)

    @test isfinite(ξ_w) || ξ_w == Inf
    @test isfinite(ξ_lw) || ξ_lw == Inf

    # Compare shape parameters
    @test isapprox(ξ_w, ξ_lw; atol=1e-8, rtol=1e-6)

    # Compare the processed weights (up to scaling): both should have same relative values
    # Normalize both to sum to 1 for a stable comparison
    w_norm = w_copy ./ sum(w_copy)
    r_norm = restored_from_log ./ sum(restored_from_log)
    @test isapprox(w_norm, r_norm; atol=1e-8, rtol=1e-6)
end
