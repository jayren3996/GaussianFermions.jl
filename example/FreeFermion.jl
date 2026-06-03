using GaussianFermions
using LinearAlgebra
using Random

#--------------------------------------------------------------------------------
# Minimal tour of the number-conserving API:
#   states & observables → unitary evolution → a monitored quantum trajectory →
#   a trajectory average.
#
# There is no ensemble runner: the trajectory step and the averaging loop live
# here, in the script, built from the library primitives (`evolve!`, `jump_rate`,
# `apply_click!`, `apply_noclick!`).
#--------------------------------------------------------------------------------

# One MCWF (quantum-jump) step: unitary, then per-channel click / no-click back-action.
function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        rate, work = jump_rate(ch, s, dt)
        if rand(rng) < rate
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)          # normalizes by default
        end
    end
    s
end

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
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=0.5) for i in 1:L]
    st = SlaterState(L=L, N=L ÷ 2, config="Z2")
    for _ in 1:100
        mcwf_step!(st, U, chans, dt)
    end
    println("monitored traj S:  ", round(entanglement_entropy(st, 1:(L ÷ 2)); digits=4))

    # --- trajectory average of the half-chain entanglement (caller owns the loop) ---
    nsteps = round(Int, tspan / dt)
    ΣS = 0.0; ΣS² = 0.0
    for _ in 1:ntraj
        traj = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for _ in 1:nsteps
            mcwf_step!(traj, U, chans, dt)
        end
        S = entanglement_entropy(traj, 1:(L ÷ 2))
        ΣS += S; ΣS² += S^2
    end
    meanS = ΣS / ntraj
    stdS = sqrt(max(ΣS² / ntraj - meanS^2, 0.0))
    println("ensemble ⟨S⟩(t=$tspan): ", round(meanS; digits=4),
            " ± ", round(stdS / sqrt(ntraj); digits=4))
    (mean=meanS, std=stdS, ntraj=ntraj)
end

if abspath(PROGRAM_FILE) == @__FILE__
    freefermion_demo()
end
