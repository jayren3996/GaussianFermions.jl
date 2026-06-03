# Measurement-Induced Entanglement Transition

Continuously monitoring a free-fermion chain competes with its unitary spreading of
entanglement. At weak monitoring, entanglement grows with subsystem size (a log-law);
at strong monitoring, repeated measurements pin the state and entanglement saturates
(an area law). Because entanglement is **trajectory-nonlinear**, the steady-state
entropy is a *trajectory average* — it cannot be obtained from the deterministic
Lindblad solver.

This page uses the quantum-jump (MCWF) unraveling of site occupation monitoring at
rate `γ`; you own the loop:

```julia
using GaussianFermions, LinearAlgebra, Random

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
```

## Half-chain entropy vs monitoring rate

```julia
for γ in (0.1, 0.5, 1.0, 2.0)
    println("γ = $γ\t⟨S(L/2)⟩ ≈ ", round(steady_entropy(8; γ=γ); digits=3))
end
```

```text
γ = 0.1   ⟨S(L/2)⟩ ≈ 2.968
γ = 0.5   ⟨S(L/2)⟩ ≈ 2.165
γ = 1.0   ⟨S(L/2)⟩ ≈ 1.472
γ = 2.0   ⟨S(L/2)⟩ ≈ 0.746
```

The steady-state entanglement falls monotonically as monitoring strengthens.

## Subsystem-size scaling

The cleaner signature is how the entropy grows with region size `ℓ`. At weak
monitoring `S(ℓ)` grows steadily with `ℓ`; at strong monitoring it grows far more
slowly, consistent with an approach to area-law saturation:

```julia
for (label, γ) in (("weak  ", 0.2), ("strong", 2.0))
    Ss = [round(steady_entropy(ℓ; L=24, γ=γ, ntraj=16); digits=3) for ℓ in 2:2:12]
    println("$label (γ=$γ): ", Ss)
end
```

```text
weak   (γ=0.2): [1.17, 2.091, 2.85, 3.43, 3.74, 3.802]
strong (γ=2.0): [0.542, 0.687, 0.736, 0.801, 0.864, 0.85]
```

!!! note "Illustrative sizes"
    These use small `L`, `ntraj`, and evolution times so they run quickly. A genuine
    finite-size-scaling study needs larger systems, more trajectories, and several
    `L` to locate the critical `γ`. The full run is in
    [`example/Transition.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/Transition.jl).

See [Mutual & Tripartite Information](monitored-mutual-information.md) for the
information-theoretic order parameter and [Monitoring Protocols](monitoring-protocols.md)
for how the unraveling choice affects the entanglement.
