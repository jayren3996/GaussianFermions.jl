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

## Majorana/BdG Foundation

General quadratic (BdG/Majorana) Gaussian states use `MajoranaState`, which stores
the real antisymmetric Majorana covariance matrix. `BdGHamiltonian` evolves this
covariance under dense unitary dynamics.

```julia
using GaussianFermions

state = MajoranaState([1, 0, 1, 0])
A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.3
B[2, 1] = -0.3

H = BdGHamiltonian(A, B)
evolve!(state, H, 0.5)

normal_correlation(state)
anomalous_correlation(state)
```

Correlations follow the standard convention `Cᵢⱼ = ⟨c⁺ᵢcⱼ⟩`, `Fᵢⱼ = ⟨cᵢcⱼ⟩` with
covariance `Γ_ab = (i/2)⟨[ωₐ,ω_b]⟩` (verified against exact diagonalization). Ground
and thermal states of a `BdGHamiltonian` come from Bogoliubov/Nambu diagonalization:

```julia
H = BdGHamiltonian(A, B)
gs = groundstate(H)              # BCS ground state as a MajoranaState
ε  = quasiparticle_energies(H)   # Bogoliubov spectrum (≥ 0)
ρβ = thermalstate(H; β=2.0)      # Gibbs state; β → ∞ recovers groundstate(H)
```

Open-system BdG dynamics use `MajoranaLindblad`, the covariance-matrix analogue of
`CorrelationLindblad`. It supports particle loss/gain, occupation dephasing, and —
unlike the number-conserving solver — general pairing (number-non-conserving) baths
via raw Majorana jump vectors.

```julia
using GaussianFermions

H = hopping(4; pbc=true)
lind = MajoranaLindblad(H; loss_ops=[sqrt(0.3) * ComplexF64[1, 0, 0, 0]],
                        dephasing_ops=[(ComplexF64[1, 1, 0, 0] / sqrt(2), 0.5)])

state = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
evolve!(state, lind, 1.0)
density(state)

normal_correlation(state)
anomalous_correlation(state)   # nonzero once a pairing bath acts
```

On number-conserving channels `MajoranaLindblad` reproduces `CorrelationLindblad`
to machine precision (see `example/Dephase.jl`).

## Quantum Trajectories

Monitored dynamics are built from **low-level primitives** — there is no trajectory
runner. You own the loop: evolve a step, decide measurements/jumps, accumulate
observables. This keeps the dynamics explicit and tunable per script. The primitives
(for both `SlaterState` and `MajoranaState`):

- `measure!(s, i)` — projective occupation measurement (Born statistics, collapse).
- `jump_rate(s, ℓ)` / `apply_click!(s, ℓ)` / `apply_noclick!(s, jumps, dt)` — linear
  quantum-jump (MCWF) building blocks; for `SlaterState` the channel forms
  `jump_rate(ch, s, dt)` / `apply_click!(ch, s, work)` / `apply_noclick!(ch, s, dt)`.
- `weak_measure!(s, i, α)` — the exact weak-measurement filter `exp(α nᵢ)` for
  continuous (quantum-state-diffusion) monitoring.

**Projective monitoring** of a `MajoranaState` and a hand-rolled trajectory average:

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

Projective measurement of `nᵢ` at rate `γ` averages to the dephasing Lindblad
`D[√(2γ) nᵢ]`, so the trajectory average reproduces `MajoranaLindblad`.

**Dissipative (loss/gain/pairing) jumps** — a Monte-Carlo wave-function (MCWF) step
draws a click (the conditional Gaussian state) or applies the effective non-Hermitian
no-click drift. `loss_jump`/`gain_jump`/`majorana_jump` build the Majorana jump vector
`ℓ`; pairing jumps generate anomalous correlations:

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

**Continuous (QSD) monitoring** applies the exact weak filter `exp(αᵢ nᵢ)` with
`αᵢ = δWᵢ + (2⟨nᵢ⟩ − 1)γᵢ dt`:

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
`D[√γ nᵢ]` (verified against the `SlaterState` filter and exact diagonalization,
including paired states). See `example/FreeFermion.jl`, `example/MI.jl`, and
`example/Dephase.jl` for complete trajectory + averaging loops.
