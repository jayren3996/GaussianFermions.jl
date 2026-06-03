# Quantum Trajectories

Monitored dynamics are built from **low-level primitives** — there is no built-in
trajectory runner. You own the loop: evolve a step, decide measurements / jumps, and
accumulate observables. This keeps the dynamics explicit and tunable per script.

The primitives work for both `SlaterState` and `MajoranaState`:

- **Projective occupation measurement** (Born statistics, collapse):
  `measure!(s::MajoranaState, i)` measures site `i`;
  `measure!(qm::QuasiMode, s::SlaterState)` measures a quasi-mode of a `SlaterState`.
- **Quantum-jump (MCWF) building blocks:** `jump_rate(s, ℓ)`, `apply_click!(s, ℓ)`,
  `apply_noclick!(s, jumps, dt)`. For `SlaterState` the channel forms are
  `jump_rate(ch, s, dt)`, `apply_click!(ch, s, work)`, `apply_noclick!(ch, s, dt)`.
- **Continuous (QSD) weak measurement:** `weak_measure!(s, i, α)` applies the exact
  weak-measurement filter ``e^{\alpha n_i}``.

The jump vectors are built with `loss_jump`, `gain_jump`, and `majorana_jump`.

## Projective monitoring

Projective measurement of ``n_i`` at rate ``\gamma`` averages to the dephasing Lindblad
``\mathcal{D}[\sqrt{2\gamma}\, n_i]``, so a trajectory average reproduces
`MajoranaLindblad`. A hand-rolled trajectory average of half-chain entanglement:

```julia
using GaussianFermions, Random

H = BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(8; pbc=true).h)))
U = propagator(H, 0.05)
rng = Xoshiro(1); ntraj = 200; ΣS = 0.0
for _ in 1:ntraj
    s = MajoranaState(CorrelationState(SlaterState(L=8, N=4, config="Z2")))
    for _ in 1:40
        evolve!(s, U)
        for i in 1:8
            rand(rng) < 0.5 * 0.05 && measure!(s, i; rng)   # monitor each site at rate 0.5
        end
    end
    global ΣS += entanglement_entropy(s, 1:4)
end
ΣS / ntraj   # trajectory-averaged half-chain entanglement
```

## Dissipative (loss / gain / pairing) jumps

A Monte-Carlo wave-function (MCWF) step either draws a click (the conditional Gaussian
state) or applies the effective non-Hermitian no-click drift. Pairing jumps generate
anomalous correlations:

```julia
using GaussianFermions, Random

U = propagator(BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(4; pbc=true).h))), 0.01)
jumps = [loss_jump(sqrt(0.4) * ComplexF64[1, 0, 0, 0]),
         gain_jump(sqrt(0.25) * ComplexF64[0, 0, 1, 0])]
rng = Xoshiro(1)
s = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
for _ in 1:150
    evolve!(s, U)
    rates = [jump_rate(s, ℓ) for ℓ in jumps]
    if rand(rng) < sum(rates) * 0.01
        apply_click!(s, jumps[argmax(rand(rng) .< cumsum(rates) ./ sum(rates))])
    else
        apply_noclick!(s, jumps, 0.01)
    end
end
density(s)   # averaged over trajectories, matches MajoranaLindblad(H; loss_ops=…, gain_ops=…)
```

## Continuous (QSD) monitoring

Continuous monitoring applies the exact weak filter ``e^{\alpha_i n_i}`` with
``\alpha_i = \delta W_i + (2\langle n_i\rangle - 1)\gamma_i\, dt``:

```julia
using GaussianFermions, Random

U = propagator(BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(4; pbc=true).h))), 0.01)
rng = Xoshiro(1); γ = 0.7
s = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
for _ in 1:100
    evolve!(s, U)
    for i in 1:4
        α = randn(rng) * sqrt(γ * 0.01) + (2 * density(s, i) - 1) * γ * 0.01
        weak_measure!(s, i, α)
    end
end
density(s)
```

The `MajoranaState` `weak_measure!` matches the number-conserving `SlaterState` one to
machine precision, and averaging over the noise reproduces the dephasing Lindblad
``\mathcal{D}[\sqrt{\gamma}\, n_i]`` (verified against exact diagonalization, including
paired states).

## Complete examples

See `example/FreeFermion.jl`, `example/MI.jl`, and `example/Dephase.jl` for full
trajectory + averaging loops.

The trajectory primitives are documented in the
[Dynamics API reference](../reference/dynamics.md).
