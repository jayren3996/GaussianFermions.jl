# Quantum Trajectories

GaussianFermions.jl exposes trajectory primitives, not a trajectory runner. A script
owns the loop: evolve, draw a measurement or jump, update the Gaussian state, and
accumulate the observables relevant to the calculation.

This design keeps the stochastic protocol visible. It also avoids baking in one
choice of sampling, batching, or data collection.

## Projective Occupation Measurements

For `MajoranaState`, `measure!(s, i; rng)` measures the site occupation ``n_i`` with
Born probability and updates the covariance by Gaussian conditioning.

```julia
using GaussianFermions, Random

rng = Xoshiro(1)
s = MajoranaState(CorrelationState(SlaterState(L=8, N=4, config="Z2")))
outcome = measure!(s, 3; rng)
```

For `SlaterState`, projective measurement is expressed in terms of a `QuasiMode`:

```julia
qm = QuasiMode([3], ComplexF64[1], 8)
outcome = measure!(qm, s; rng)
```

Projective monitoring of ``n_i`` at rate ``\gamma`` averages to a dephasing
Lindblad channel with jump ``\sqrt{2\gamma}\,n_i``.

## Number-Conserving MCWF Steps

Number-conserving trajectory channels are typed objects:

```julia
channels = [dephasing(i, L; γ=0.5) for i in 1:L]
```

The channel form of `jump_rate` returns a click probability over the step `dt` plus
work data reused by `apply_click!`.

```julia
function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        probability, work = jump_rate(ch, s, dt)
        if rand(rng) < probability
            apply_click!(ch, s, work)
            normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end
```

`apply_noclick!` applies the non-unitary no-click back-action and normalizes by
default.

## Majorana Linear Jumps

For `MajoranaState`, linear jumps are built as Majorana jump vectors:

```julia
jumps = [
    loss_jump(sqrt(0.4) * ComplexF64[1, 0, 0, 0]),
    gain_jump(sqrt(0.25) * ComplexF64[0, 0, 1, 0]),
]
```

Here `jump_rate(s, ell)` returns an instantaneous rate, not a probability. The caller
multiplies by `dt` when drawing events.

```julia
rates = [jump_rate(s, ell) for ell in jumps]
if rand(rng) < sum(rates) * dt
    apply_click!(s, jumps[argmax(rand(rng) .< cumsum(rates) ./ sum(rates))])
else
    apply_noclick!(s, jumps, dt)
end
```

Pairing jumps can generate nonzero anomalous correlations, so this path is useful
when the monitored or dissipative protocol is not number-conserving.

## Weak Measurements

`weak_measure!(s, i, α)` applies the exact Gaussian filter ``\exp(\alpha n_i)``.
For a QSD-style occupation-monitoring step one common update is

```math
\alpha_i = \delta W_i + (2\langle n_i\rangle - 1)\gamma_i\,dt .
```

```julia
for i in 1:L
    α = randn(rng) * sqrt(γ * dt) + (2 * density(s, i) - 1) * γ * dt
    weak_measure!(s, i, α)
end
```

See [Free-Fermion Chain](../examples/free-fermion-chain.md),
[Trajectories vs Lindblad](../examples/trajectories-vs-lindblad.md), and
[Monitored Mutual Information](../examples/monitored-mutual-information.md) for
complete loops.
