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

Monitored BdG quantum trajectories use `measure!` (projective occupation collapse of
a `MajoranaState`) and `step!` (unitary evolution + projective monitoring), which plug
into the existing `ensemble` runner:

```julia
using GaussianFermions

H = BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(8; pbc=true).h)))
res = ensemble(() -> MajoranaState(CorrelationState(SlaterState(L=8, N=4, config="Z2"))),
               H, [(i, 0.5) for i in 1:8];          # projectively monitor each site at rate 0.5
               ntraj=200, tspan=2.0, dt=0.05,
               observables=(n = density, S = s -> entanglement_entropy(s, 1:4)))
res.mean.S[end]   # trajectory-averaged half-chain entanglement
```

Projective measurement of `nᵢ` at rate `γ` averages to the dephasing Lindblad
`D[√(2γ) nᵢ]`, so the trajectory ensemble reproduces `MajoranaLindblad`.

For dissipative (loss/gain/pairing) baths, `MajoranaJumps` packages the linear jump
operators into a Monte-Carlo wave-function (MCWF) unraveling: each step draws a click
(rendering the conditional Gaussian state) or follows the effective non-Hermitian
no-click drift. Unlike projective monitoring this handles number-non-conserving
(pairing) jumps and generates anomalous correlations.

```julia
using GaussianFermions

H  = BdGHamiltonian(QuadraticHamiltonian(Matrix(hopping(4; pbc=true).h)))
J  = MajoranaJumps(4; loss_ops=[sqrt(0.4) * ComplexF64[1, 0, 0, 0]],
                      gain_ops=[sqrt(0.25) * ComplexF64[0, 0, 1, 0]])
res = ensemble(() -> MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2"))),
               H, J; ntraj=3000, tspan=1.5, dt=0.01, observables=(n = density,))
res.mean.n[end]   # matches MajoranaLindblad(H; loss_ops=…, gain_ops=…)
```

The MCWF ensemble reproduces the deterministic `MajoranaLindblad` built from the same
jump operators (density and anomalous correlations, verified).
