using GaussianFermions, LinearAlgebra, Random

#--------------------------------------------------------------------------------
# Measurement-induced entanglement transition for monitored free fermions.
# Hopping chain + projective-style occupation monitoring at rate γ, unravelled as a
# quantum-jump (MCWF) trajectory. Entanglement is trajectory-nonlinear, so the
# steady-state entropy is a TRAJECTORY AVERAGE — it cannot be read from the
# deterministic Lindblad solver. As γ grows, ⟨S⟩ crosses over from a log-law (weak
# monitoring) to an area law (strong monitoring).
#--------------------------------------------------------------------------------

function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        p, work = jump_rate(ch, s, dt)
        if rand(rng) < p
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end

"Late-time, trajectory-averaged entanglement entropy of region 1:ℓ."
function steady_entropy(ℓ; L=16, γ=1.0, dt=0.1, tspan=12.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4
    Σ = 0.0; nsamp = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            k > nsteps - tail && (Σ += entanglement_entropy(s, 1:ℓ); nsamp += 1)
        end
    end
    Σ / nsamp
end

function main()
    println("half-chain entropy vs monitoring rate (L=16):")
    for γ in (0.1, 0.5, 1.0, 2.0)
        println("  γ = $γ\t⟨S(L/2)⟩ ≈ ", round(steady_entropy(8; γ=γ); digits=3))
    end
    println("subsystem-size scaling S(ℓ), L=24:")
    for (label, γ) in (("weak  ", 0.2), ("strong", 2.0))
        Ss = [round(steady_entropy(ℓ; L=24, γ=γ, ntraj=16); digits=3) for ℓ in 2:2:12]
        println("  $label (γ=$γ): ", Ss)
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
