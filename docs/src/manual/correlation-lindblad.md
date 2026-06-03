# Correlation-Matrix Lindblad

For number-conserving open systems, `CorrelationLindblad` evolves the single-particle
correlation matrix ``C`` under a quadratic Lindblad master equation with linear (loss /
gain) and quadratic (dephasing) jump operators. The dynamics stay closed at the level of
``C``, so the cost is polynomial in system size.

## Building a Lindbladian

```julia
using GaussianFermions

L = 16
H = hopping(L; pbc=true)

loss1 = zeros(ComplexF64, L); loss1[1] = sqrt(0.2)   # particle loss at site 1
lind  = CorrelationLindblad(H; loss_ops=[loss1])
```

The `loss_ops` and `gain_ops` keywords take **Lindblad amplitude vectors**: for rate
``\gamma`` acting on a normalized mode ``v``, pass ``\sqrt{\gamma}\,v``. The Hamiltonian
`H` supplies the coherent part.

## Time evolution

```julia
state = CorrelationState(SlaterState(L=L, N=8, config="Z2"))
evolve!(state, lind, 1.0)
density(state)
```

The deterministic solver is dense and intended for finite systems. The right-hand side
of the master equation is exposed as `lindblad_rhs` if you want to plug it into a custom
integrator.

## Steady states

For a finite system with a unique attractive steady state, solve for it directly with
`steadystate` instead of integrating to long times:

```julia
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                         loss_ops=[ComplexF64[sqrt(0.6)]],
                                         gain_ops=[ComplexF64[sqrt(0.2)]]))
density(steady)   # ≈ [0.25]
```

## Relation to other solvers

`CorrelationLindblad` is the number-conserving specialisation of
[`MajoranaLindblad`](majorana-bdg.md), which additionally handles pairing
(number-non-conserving) baths. On number-conserving channels the two agree to machine
precision (see `example/Dephase.jl`). The same dynamics can also be unravelled into
[quantum trajectories](trajectories.md).

See the [Dynamics API reference](../reference/dynamics.md).
