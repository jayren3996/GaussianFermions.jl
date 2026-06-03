# Lindblad Dynamics

GaussianFermions.jl provides deterministic Lindblad evolution at the covariance level.
The solver you choose depends on whether the dynamics preserve particle number.

| Dynamics | State | Solver |
| --- | --- | --- |
| Number-conserving loss/gain/dephasing | `CorrelationState` | `CorrelationLindblad` |
| BdG dynamics with pairing baths | `MajoranaState` | `MajoranaLindblad` |

Both solvers are dense finite-system tools. They are meant for moderate system sizes
where forming and exponentiating the affine generator is acceptable.

## Number-Conserving Lindblad Dynamics

`CorrelationLindblad` evolves the normal correlation matrix ``C``. Loss and gain
operators are supplied as amplitude vectors: for rate ``\gamma`` acting on a
normalized mode ``v``, pass ``\sqrt{\gamma}v``.

```@example lindblad
using GaussianFermions

L = 8
H = hopping(L; pbc=true)

loss1 = zeros(ComplexF64, L)
loss1[1] = sqrt(0.2)

lind = CorrelationLindblad(H; loss_ops=[loss1])
state = CorrelationState(SlaterState(L=L, N=4, config="Z2"))
evolve!(state, lind, 1.0)
round.(density(state); digits=3)
```

The same constructor also accepts the number-conserving channel objects from the
trajectory API:

```julia
channels = [dephasing(i, L; γ=0.5) for i in 1:L]
lind = CorrelationLindblad(H, channels)
```

The right-hand side is exposed as `lindblad_rhs` for custom integrators.

## Steady States

For finite systems with an attractive steady state, `steadystate` solves the affine
problem directly:

```@example lindblad
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                         loss_ops=[ComplexF64[sqrt(0.6)]],
                                         gain_ops=[ComplexF64[sqrt(0.2)]]))
round.(density(steady); digits=3)
```

## Majorana Lindblad Dynamics

`MajoranaLindblad` evolves ``\Gamma`` and supports all number-conserving channels
above, plus pairing baths through Majorana-linear jump vectors.

```@example lindblad
H4 = hopping(4; pbc=true)
lind4 = MajoranaLindblad(H4;
    loss_ops=[sqrt(0.3) * ComplexF64[1, 0, 0, 0]],
    dephasing_ops=[(ComplexF64[1, 1, 0, 0] / sqrt(2), 0.5)],
)

maj = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
evolve!(maj, lind4, 0.5)
round.(density(maj); digits=3)
```

For number-conserving channels, `MajoranaLindblad` reproduces `CorrelationLindblad`
up to numerical tolerance. Use the Majorana solver when the state or bath can generate
anomalous correlations.

## Relation To Trajectories

Projective monitoring, MCWF jumps, and weak-measurement updates can unravel the same
Lindblad dynamics. GaussianFermions.jl exposes those conditional updates as low-level
primitives; see [Quantum Trajectories](trajectories.md) and
[Trajectories vs Lindblad](../examples/trajectories-vs-lindblad.md).
