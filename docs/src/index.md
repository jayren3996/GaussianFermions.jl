# GaussianFermions.jl

*Simulation of fermionic Gaussian states under unitary, Lindblad, and quantum-trajectory
(monitored) dynamics.*

GaussianFermions.jl provides composable primitives for the efficient classical
simulation of free-fermion systems. It covers two complementary representations:

- **Number-conserving (U(1)) states** — `SlaterState` and `CorrelationState`, built
  from the single-particle correlation matrix ``C_{ij} = \langle c_i^\dagger c_j\rangle``.
- **General quadratic (BdG / Majorana) states** — `MajoranaState`, built from the real
  antisymmetric Majorana covariance matrix, supporting pairing
  (number-non-conserving) dynamics.

On top of these states the package implements:

- **Unitary dynamics** via `QuadraticHamiltonian` / `BdGHamiltonian` and `propagator`.
- **Deterministic Lindblad dynamics** via `CorrelationLindblad` and `MajoranaLindblad`,
  including particle loss/gain, dephasing, and pairing baths.
- **Quantum trajectories** built from low-level primitives — projective measurement,
  quantum-jump (MCWF), and continuous (QSD) weak measurement — so you own the
  trajectory loop and can tune it per script.
- **Observables** — densities, particle-number statistics, entanglement entropies and
  spectra, mutual / tripartite information.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

## Quick example

```julia
using GaussianFermions

L = 16
H = hopping(L; pbc=true)

loss1 = zeros(ComplexF64, L); loss1[1] = sqrt(0.2)
lind = CorrelationLindblad(H; loss_ops=[loss1])

state = CorrelationState(SlaterState(L=L, N=8, config="Z2"))
evolve!(state, lind, 1.0)
density(state)
```

## Where to go next

- New here? Start with [Getting Started](getting-started.md).
- The **Manual** walks through each layer: [States](manual/states.md),
  [Hamiltonians & Time Evolution](manual/hamiltonians.md),
  [Observables](manual/observables.md),
  [Correlation-Matrix Lindblad](manual/correlation-lindblad.md),
  [Majorana / BdG Foundation](manual/majorana-bdg.md), and
  [Quantum Trajectories](manual/trajectories.md).
- The [API Reference](reference/overview.md) documents every exported function and type.

## Conventions

Single-particle correlations follow the standard convention
``C_{ij} = \langle c_i^\dagger c_j\rangle``, ``F_{ij} = \langle c_i c_j\rangle``, with the
Majorana covariance matrix ``\Gamma_{ab} = \tfrac{i}{2}\langle[\omega_a, \omega_b]\rangle``.
All conventions are verified against exact diagonalization.
