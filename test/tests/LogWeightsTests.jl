using Test

@testset "Log Weights Handling" begin
    # Simple synthetic example where importance ratios are obtained in log-space
    # Create a vector of log-weights with a clear tail
    log_w = [-10.0, -9.0, -8.0, -7.0, -6.0, -5.0, -4.0, -3.0, -2.0, 0.0]
    raw = copy(log_w)

    # Call psis! on a copy with log_weights=true and ensure it does not error
    new = copy(raw)
    ξ = psis!(new; log_weights=true)
    @test isfinite(ξ) || ξ == Inf

    # Ensure that returned new has been processed and remains in log-space (max at 0)
    @test maximum(new) ≈ 0.0

    # Also test non-log path for the same data exponentiated
    ratios = exp.(log_w .- maximum(log_w))
    new2 = copy(ratios)
    ξ2 = psis!(new2; log_weights=false)
    @test isfinite(ξ2) || ξ2 == Inf
    @test maximum(new2) ≤ 1.0 + 1e-12

    # Check the vector psis() forwards kwargs correctly by calling psis on a vector
    vec = copy(log_w)
    new_vec, ξv = psis(vec; log_weights=true)
    @test isfinite(ξv) || ξv == Inf
    @test maximum(new_vec) ≈ 0.0
end
