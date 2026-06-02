# GaussianFermions.jl

Simulation of fermionic Gaussian states under unitary, Lindblad, and quantum-trajectory
(monitored) dynamics. Supports number-conserving free-fermion states as well as general
quadratic/Majorana states.

## Installation

```julia
pkg> add https://github.com/jayren3996/GaussianFermions.jl
```

## Deterministic Correlation-Matrix Lindblad Dynamics

```julia
using GaussianFermions

L = 16
H = hopping(L; pbc=true)

loss1 = zeros(ComplexF64, L)
loss1[1] = sqrt(0.2)

lind = CorrelationLindblad(H; loss_ops=[loss1])
state = CorrelationState(SlaterState(L=L, N=8, config="Z2"))

evolve!(state, lind, 1.0)
density(state)
```

Low-level `loss_ops` and `gain_ops` are Lindblad amplitude vectors: for rate γ on normalized mode v, pass `sqrt(γ) * v`.
The current deterministic solver is dense and intended for finite systems.

For finite systems with a unique attractive state, solve directly:

```julia
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                        loss_ops=[ComplexF64[sqrt(0.6)]],
                                        gain_ops=[ComplexF64[sqrt(0.2)]]))
density(steady)  # ≈ [0.25]
```
