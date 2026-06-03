# Deterministic Lindblad

This workflow evolves a number-conserving correlation matrix under local particle
loss. The loss vector is an amplitude vector: rate ``\gamma`` on normalized mode `v`
is passed as ``\sqrt{\gamma}v``.

```@example deterministic_lindblad
using GaussianFermions

L = 10
H = hopping(L; pbc=false)

loss_left = zeros(ComplexF64, L)
loss_left[1] = sqrt(0.4)

lind = CorrelationLindblad(H; loss_ops=[loss_left])
state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))

for _ in 1:20
    evolve!(state, lind, 0.05)
end

round.(density(state); digits=3)
```

For steady states, solve the affine problem directly:

```@example deterministic_lindblad
steady = steadystate(CorrelationLindblad(zeros(ComplexF64, 1, 1);
                                         loss_ops=[ComplexF64[sqrt(0.6)]],
                                         gain_ops=[ComplexF64[sqrt(0.2)]]))
round.(density(steady); digits=3)
```
