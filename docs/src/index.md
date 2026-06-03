```@raw html
<div align="center">
<img src="assets/logo.svg" alt="GaussianFermions.jl logo" width="180"/>
</div>
```

# GaussianFermions.jl

GaussianFermions.jl simulates finite fermionic Gaussian systems under quadratic
Hamiltonian dynamics, Gaussian Lindblad dynamics, and monitored trajectory updates.
It is written for researchers who already know the physics and need a precise Julia
interface with explicit conventions.

The package has two state representations:

- `SlaterState` stores a pure number-conserving Gaussian state by its occupied
  orbitals.
- `CorrelationState` stores a possibly mixed number-conserving Gaussian state by
  ``C_{ij} = \langle c_i^\dagger c_j\rangle``.
- `MajoranaState` stores a general Gaussian state, including pairing, by the
  Majorana covariance matrix ``\Gamma``.

## Choosing A Representation

| Modeling need | Use |
| --- | --- |
| Pure fixed-particle states | `SlaterState` |
| Mixed ``U(1)``-symmetric states | `CorrelationState` |
| Pairing, BdG dynamics, anomalous correlators | `MajoranaState` |
| Number-conserving quadratic Hamiltonians | `QuadraticHamiltonian` |
| BdG Hamiltonians with pairing | `BdGHamiltonian` |
| Deterministic ``U(1)`` open dynamics | `CorrelationLindblad` |
| Deterministic BdG open dynamics | `MajoranaLindblad` |
| Monitored trajectories | compose the low-level primitives in your loop |

The last row is intentional. GaussianFermions.jl does not hide trajectory sampling
behind an ensemble runner. You evolve a state, draw measurements or jumps, update the
state, and accumulate the observables needed by your calculation.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

```julia
using GaussianFermions
```

## A Minimal Run

```@example home
using GaussianFermions

L = 8
H = hopping(L; pbc=true)
state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="Z2"))

evolve!(state, H, 0.5)
round.(density(state); digits=3)
```

## Where To Start

- Read [Conventions](manual/conventions.md) before comparing signs or correlations
  with another codebase.
- Read [States](manual/states.md) and
  [Hamiltonians & Time Evolution](manual/hamiltonians.md) to model a closed system.
- Read [Lindblad Dynamics](manual/lindblad.md) for deterministic open-system
  evolution.
- Read [Quantum Trajectories](manual/trajectories.md) for monitored dynamics.
- Use the [Examples](examples/free-fermion-chain.md) for complete script patterns.
- Use the [API Reference](reference/overview.md) for signatures and mutation
  behavior.
