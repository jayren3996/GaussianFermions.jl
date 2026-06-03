# Getting Started

This page walks through the core workflow: prepare a state, evolve it, and measure
observables. Every example is self-contained.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

```julia
using GaussianFermions
```

## 1. Prepare a state

A number-conserving free-fermion state is a `SlaterState`. Here a half-filled chain in a
`Z2` (alternating-occupation) product configuration:

```julia
L = 8
s = SlaterState(L=L, N=4, config="Z2")
density(s)            # site occupations
```

For correlation-matrix dynamics, wrap it in a `CorrelationState`:

```julia
state = CorrelationState(s)
```

## 2. Build a Hamiltonian

Free-fermion (single-particle) Hamiltonians are `QuadraticHamiltonian`s. Convenience
builders construct common terms:

```julia
H = hopping(L; pbc=true)          # nearest-neighbour hopping, periodic
```

## 3. Evolve

`evolve!` advances a state in place. For unitary dynamics with a `QuadraticHamiltonian`:

```julia
evolve!(state, H, 0.5)            # evolve for time t = 0.5
density(state)
```

For repeated steps, precompute a `propagator`:

```julia
U = propagator(H, 0.05)
for _ in 1:20
    evolve!(state, U)
end
```

## 4. Measure observables

```julia
density(state)                    # ⟨nᵢ⟩ per site
particle_number(state)            # total ⟨N⟩
entanglement_entropy(state, 1:4)  # half-chain entanglement entropy
```

## 5. Add dissipation

Open-system (Lindblad) dynamics for number-conserving channels use
`CorrelationLindblad`:

```julia
loss1 = zeros(ComplexF64, L); loss1[1] = sqrt(0.2)   # √γ · v for rate γ on mode v
lind  = CorrelationLindblad(H; loss_ops=[loss1])

state = CorrelationState(SlaterState(L=L, N=4, config="Z2"))
evolve!(state, lind, 1.0)
density(state)
```

## Next steps

- [States](manual/states.md) — `SlaterState`, `CorrelationState`, `MajoranaState`, and
  conversions between them.
- [Correlation-Matrix Lindblad](manual/correlation-lindblad.md) and
  [Majorana / BdG Foundation](manual/majorana-bdg.md) — open-system dynamics.
- [Quantum Trajectories](manual/trajectories.md) — monitored / measurement dynamics.
