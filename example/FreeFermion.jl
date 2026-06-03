using GaussianFermions
using LinearAlgebra

#--------------------------------------------------------------------------------
# Minimal tour of the number-conserving API:
#   states & observables → unitary evolution → a monitored quantum trajectory →
#   an ensemble average.
#--------------------------------------------------------------------------------
function freefermion_demo(; L=16, ntraj=64, tspan=5.0, dt=0.05)
    # --- states & observables ---
    s = SlaterState(L=8, N=4, config="Z2")
    println("occupation:        ", round.(density(s); digits=3))
    println("particle number:   ", particle_number(s))
    println("half-chain S:      ", round(entanglement_entropy(s, 1:4); digits=4))

    # --- unitary evolution under a hopping chain ---
    H = hopping(8; pbc=true)                 # H = Σ c†ᵢcᵢ₊₁ + h.c.
    evolve!(s, H, 1.0)
    println("after evolve, S:   ", round(entanglement_entropy(s, 1:4); digits=4),
            "  (N conserved: ", particle_number(s), ")")

    # --- a single monitored (quantum-jump) trajectory: occupation monitoring ---
    st = SlaterState(L=L, N=L ÷ 2, config="Z2")
    chans = [dephasing(i, L; γ=0.5) for i in 1:L]
    for _ in 1:100
        step!(st, hopping(L; pbc=true), chans, dt)
    end
    println("monitored traj S:  ", round(entanglement_entropy(st, 1:(L ÷ 2)); digits=4))

    # --- ensemble average of the entanglement entropy over many trajectories ---
    res = ensemble(() -> SlaterState(L=L, N=L ÷ 2, config="Z2"),
                   hopping(L; pbc=true),
                   [dephasing(i, L; γ=0.5) for i in 1:L];
                   ntraj=ntraj, tspan=tspan, dt=dt,
                   observables=(S = s -> entanglement_entropy(s, 1:(L ÷ 2)),))
    println("ensemble ⟨S⟩(t=$tspan): ", round(res.mean.S[end]; digits=4),
            " ± ", round(res.std.S[end] / sqrt(res.ntraj); digits=4))
    res
end

if abspath(PROGRAM_FILE) == @__FILE__
    freefermion_demo()
end
