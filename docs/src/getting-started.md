# Getting Started

This page walks through one full session: prepare a state, build a Hamiltonian,
evolve it, measure observables, and add dissipation. Every code block runs as part
of the documentation build.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

```julia
using GaussianFermions
```

## 1. Prepare a state

A pure number-conserving free-fermion state is a [`SlaterState`](@ref). Here a
half-filled chain in a `Z2` (alternating-occupation) product configuration:

```@example getting_started
using GaussianFermions

L = 8
s = SlaterState(L=L, N=4, config="Z2")
density(s)
```

For correlation-matrix dynamics, wrap it in a [`CorrelationState`](@ref):

```@example getting_started
state = CorrelationState(s)
particle_number(state)
```

## 2. Build a Hamiltonian

Free-fermion (single-particle) Hamiltonians are `QuadraticHamiltonian`s, with
convenience builders for the common chain terms:

```@example getting_started
H = hopping(L; pbc=true)          # nearest-neighbour hopping, periodic
nothing # hide
```

## 3. Evolve

`evolve!` advances a state in place; for repeated fixed steps, precompute a
`propagator`:

```@example getting_started
evolve!(state, H, 0.5)            # evolve for time t = 0.5
U = propagator(H, 0.05)
for _ in 1:20
    evolve!(state, U)
end
round(particle_number(state); digits=8)
```

## 4. Measure observables

```@example getting_started
(density        = round.(density(state); digits=3),
 number         = round(particle_number(state); digits=6),
 half_chain_S   = round(entanglement_entropy(state, 1:4); digits=4))
```

## 5. Add dissipation

Number-conserving open dynamics use `CorrelationLindblad`. Loss/gain inputs are
amplitude vectors: for rate `γ` on a normalized mode `v`, pass `√γ · v`.

```@example getting_started
loss1 = zeros(ComplexF64, L); loss1[1] = sqrt(0.2)   # loss at site 1, rate 0.2
lind  = CorrelationLindblad(H; loss_ops=[loss1])

open_state = CorrelationState(SlaterState(L=L, N=4, config="Z2"))
evolve!(open_state, lind, 1.0)
round.(density(open_state); digits=3)
```

## 6. Monitor a trajectory

There is no ensemble runner: you own the loop. A single quantum-jump (MCWF) step
evolves, then applies per-channel click / no-click back-action:

```@example getting_started
using Random, LinearAlgebra

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

rng = Xoshiro(1)
Um  = propagator(hopping(L; pbc=true), 0.1)
chans = [dephasing(i, L; γ=0.5) for i in 1:L]
traj = SlaterState(L=L, N=4, config="Z2")
for _ in 1:50
    mcwf_step!(traj, Um, chans, 0.1; rng)
end
round(entanglement_entropy(traj, 1:4); digits=4)
```

## Next steps

- [Conventions](manual/conventions.md) — read before comparing signs or correlations
  with another codebase.
- [States](manual/states.md) and [Hamiltonians & Time Evolution](manual/hamiltonians.md)
  — model a closed system.
- [Lindblad Dynamics](manual/lindblad.md) — deterministic open-system evolution.
- [Quantum Trajectories](manual/trajectories.md) — monitored dynamics and
  measurement-induced transitions.
- [API Reference](reference/overview.md) — signatures and mutation behavior.
