# Deterministic Lindblad

`CorrelationLindblad` evolves the number-conserving correlation matrix `C` under a
quadratic Lindblad master equation. It is dense and intended for finite systems. Loss
and gain inputs are amplitude vectors: rate `γ` on a normalized mode `v` is `√γ · v`.

## Local particle loss

```@example deterministic_lindblad
using GaussianFermions

L = 10
H = hopping(L; pbc=false)

loss_left = zeros(ComplexF64, L)
loss_left[1] = sqrt(0.4)

lind  = CorrelationLindblad(H; loss_ops=[loss_left])
state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))

for _ in 1:20
    evolve!(state, lind, 0.05)
end
round.(density(state); digits=3)
```

## From trajectory channels

The same constructor accepts the number-conserving channel objects from the
trajectory API, which is convenient when comparing a master equation with its
unravelings:

```@example deterministic_lindblad
channels = [dephasing(i, L; γ=0.5) for i in 1:L]
lind2 = CorrelationLindblad(H, channels)
nothing # hide
```

The right-hand side is exposed as `lindblad_rhs` for use in a custom integrator.

## Steady states

For a finite system with a unique attractive steady state, solve the affine problem
directly instead of integrating to long times:

```@example deterministic_lindblad
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                         loss_ops=[ComplexF64[sqrt(0.6)]],
                                         gain_ops=[ComplexF64[sqrt(0.2)]]))
round.(density(steady); digits=3)   # ≈ [0.25]
```

The balance of loss rate `0.6` and gain rate `0.2` fixes the steady occupation at
`0.2 / (0.2 + 0.6) = 0.25`.

`CorrelationLindblad` is the number-conserving specialization of `MajoranaLindblad`,
which additionally handles pairing baths; see [BdG Pairing](bdg-pairing.md). The same
dynamics can be unravelled into [trajectories](trajectories-vs-lindblad.md).
