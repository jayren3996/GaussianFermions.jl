# Free-Fermion Chain

This workflow shows the number-conserving path end to end: prepare a Slater
determinant, evolve under a hopping Hamiltonian, then run a full monitored
trajectory and average an observable over the ensemble — the loop you own.

## Closed evolution

```@example freefermion
using GaussianFermions

L = 12
H = hopping(L; pbc=true)
s = SlaterState(L=L, N=L ÷ 2, config="Z2")

S0 = entanglement_entropy(s, 1:(L ÷ 2))
evolve!(s, H, 1.0)
S1 = entanglement_entropy(s, 1:(L ÷ 2))

(initial_entropy=round(S0; digits=4),
 final_entropy=round(S1; digits=4),
 particle_number=particle_number(s))
```

For fixed time steps, precompute the propagator once:

```@example freefermion
U = propagator(H, 0.05)
for _ in 1:100
    evolve!(s, U)
end
round(particle_number(s); digits=8)   # conserved
```

## A monitored trajectory

There is no ensemble runner. One MCWF step evolves, then applies per-channel
click / no-click back-action:

```julia
using GaussianFermions, LinearAlgebra, Random

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
```

## Trajectory average

The caller owns the averaging loop too. Here, the late-time half-chain
entanglement under occupation monitoring, with a standard error from the
ensemble spread:

```julia
function ensemble_entropy(; L=16, γ=0.5, dt=0.05, tspan=5.0, ntraj=64, rng=Xoshiro(1))
    U     = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    nsteps = round(Int, tspan / dt)
    ΣS = 0.0; ΣS² = 0.0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for _ in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
        end
        S = entanglement_entropy(s, 1:(L ÷ 2))
        ΣS += S; ΣS² += S^2
    end
    mean = ΣS / ntraj
    stderr = sqrt(max(ΣS² / ntraj - mean^2, 0.0)) / sqrt(ntraj)
    (mean=round(mean; digits=4), stderr=round(stderr; digits=4))
end

ensemble_entropy()
# (mean ≈ 1.5–2.0, stderr ≈ 0.05 for these small-ensemble parameters)
```

The full script [`example/FreeFermion.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/FreeFermion.jl)
runs this at larger sizes. For the monitoring-strength dependence of the
steady-state entanglement, see the
[Measurement-Induced Transition](measurement-induced-transition.md) example.
