using GaussianFermions
using LinearAlgebra
using Random

#--------------------------------------------------------------------------------
# Measurement-induced behaviour: steady-state mutual information of two antipodal
# regions of a monitored hopping ring, as a function of monitoring strength γ.
# (Illustrative / small sizes — not a production finite-size-scaling run.)
#
# The trajectory step and the averaging loop live here; the library only supplies
# the per-step primitives (`evolve!`, `jump_rate`, `apply_click!`, `apply_noclick!`).
#--------------------------------------------------------------------------------

"""
Nodal monitoring modes dᵢ⁺ = (cᵢ₋₁⁺ + cᵢ⁺ + cᵢ₊₁⁺)/√3, one per site (PBC).
"""
function nodal_monitors(L; γ)
    V = ComplexF64[1, 1, 1] / sqrt(3)
    [OccupationMonitor(QuasiMode([mod(i-2, L)+1, i, mod(i, L)+1], V, L); γ=γ) for i in 1:L]
end

# One MCWF (quantum-jump) step under occupation monitoring.
function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        rate, work = jump_rate(ch, s, dt)
        if rand(rng) < rate
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end

"""
Late-time average mutual information I(A:B) over a monitored trajectory ensemble,
where A and B are two opposite quarter-rings.
"""
function mi_vs_gamma(; L=32, γ=1.0, dt=0.05, tspan=20.0, ntraj=32, rng=Random.default_rng())
    U = propagator(hopping(L; pbc=true), dt)
    chans = nodal_monitors(L; γ)
    A = collect(1:(L ÷ 4)); B = collect((L ÷ 2 + 1):(3L ÷ 4))
    nsteps = round(Int, tspan / dt)
    tail = max(1, nsteps ÷ 5)              # average over the last 20% of the evolution
    Σ = 0.0; cnt = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            if k > nsteps - tail
                Σ += mutual_information(s, A, B); cnt += 1
            end
        end
    end
    Σ / cnt
end

function main()
    for γ in (0.2, 0.5, 1.0, 2.0)
        I = mi_vs_gamma(; L=32, γ=γ, ntraj=24, tspan=15.0)
        println("γ = $γ\t⟨I(A:B)⟩ ≈ $(round(I; digits=4))")
    end
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
