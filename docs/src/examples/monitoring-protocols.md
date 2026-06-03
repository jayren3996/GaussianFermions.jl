# Monitoring Protocols Compared

The same Lindbladian can be unravelled into trajectories in different ways.
GaussianFermions.jl provides three occupation-monitoring protocols:

- **projective** — [`measure!`](@ref) collapses `nᵢ` with Born statistics;
- **quantum-jump (MCWF)** — `OccupationMonitor` channels with click / no-click
  back-action;
- **quantum-state diffusion (QSD)** — [`weak_measure!`](@ref) applies the continuous
  weak filter `exp(α nᵢ)`.

A key fact: **linear** observables (like the site density) agree across all three
unravelings after averaging, and match the deterministic `CorrelationLindblad`. But
**entanglement is trajectory-nonlinear** — it depends on the unraveling, so the
protocols generally give *different* entropies, even though each individually shows
monitoring-induced suppression. The transition is robust as *phenomenology*, not
value-for-value.

## Convention matching

All three target the dephasing Lindbladian `D[√γ nᵢ]`. Projective measurement of
`nᵢ` at rate `γₚ` unravels `D[√(2γₚ) nᵢ]`, so to compare against `D[√γ nᵢ]` use
projective rate `γ/2` while the channel and QSD rates are `γ`.

## Linear observable: density agrees

```julia
using GaussianFermions, LinearAlgebra, Random

# (mcwf_step!, lindblad_density, mcwf_density, qsd_density, projective_density:
#  see example/Protocols.jl — reproduced there in full.)

ref = lindblad_density()
println("Lindblad   density[1:4] = ", round.(ref[1:4]; digits=3))
println("MCWF       ‖Δ‖ = ", round(norm(mcwf_density() - ref); digits=3))
println("QSD        ‖Δ‖ = ", round(norm(qsd_density() - ref); digits=3))
println("Projective ‖Δ‖ = ", round(norm(projective_density() - ref); digits=3))
```

```text
Lindblad   density[1:4] = [0.905, 0.896, 0.827, 0.746]
MCWF       ‖Δ‖ = 0.053
QSD        ‖Δ‖ = 0.040
Projective ‖Δ‖ = 0.038
```

All three unravelings reproduce the deterministic density; the residuals are pure
`1/√ntraj` sampling error.

## Nonlinear observable: entanglement differs

Accumulating `entanglement_entropy(s, 1:L÷2)` instead of density (on an `L=16` ring)
gives **different** trajectory-averaged entropies for the three protocols, even though
they unravel the same Lindbladian:

```text
γ=0.5  MCWF=2.165  QSD=1.812  projective=2.122
γ=1.0  MCWF=1.472  QSD=1.272  projective=1.404
γ=2.0  MCWF=0.746  QSD=0.677  projective=0.839
```

The three numbers never coincide, and their order is not even fixed — here QSD is
consistently the lowest, while MCWF and projective swap as `γ` grows. Each protocol
still shows monitoring-induced suppression (every column falls as `γ` increases).
This is expected: the unraveling encodes *which* measurement record is kept, and
entanglement is conditioned on that record, so — unlike the averaged density — it is
not unraveling-invariant.

The full runnable comparison is in
[`example/Protocols.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/Protocols.jl).
See [Trajectories vs Lindblad](trajectories-vs-lindblad.md) for the linear-observable
agreement in more detail, and [Measurement-Induced Transition](measurement-induced-transition.md)
for the entanglement transition itself.
