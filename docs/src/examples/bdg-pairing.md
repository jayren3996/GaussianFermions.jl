# BdG Pairing

Use `MajoranaState` and `BdGHamiltonian` whenever the Hamiltonian or a bath can
generate anomalous correlations `Fᵢⱼ = ⟨cᵢcⱼ⟩`. `A` is the Hermitian hopping block,
`B` the antisymmetric pairing block.

## Unitary pairing dynamics

Starting from the empty state, a pairing Hamiltonian builds up anomalous
correlations:

```@example bdg_pairing
using GaussianFermions, LinearAlgebra

A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.35; B[2, 1] = -0.35
B[3, 4] = -0.2; B[4, 3] = 0.2

H = BdGHamiltonian(A, B)
s = MajoranaState([0, 0, 0, 0])

evolve!(s, H, 0.75)
(density        = round.(density(s); digits=3),
 anomalous_norm = round(norm(anomalous_correlation(s)); digits=4))
```

## Ground and thermal states

Ground and thermal states come from Nambu/Bogoliubov diagonalization:

```@example bdg_pairing
eps = quasiparticle_spectrum(H)
gs  = groundstate(H)
ρβ  = thermalstate(H; β=2.0)

(min_quasiparticle_energy = round(minimum(eps); digits=4),
 ground_particle_number   = round(particle_number(gs); digits=4),
 thermal_particle_number  = round(particle_number(ρβ); digits=4))
```

## A pairing bath

Unlike the number-conserving solver, `MajoranaLindblad` accepts pairing
(number-non-conserving) baths through Majorana-linear jump vectors, which generate
nonzero anomalous correlations even from a number-conserving initial state:

```@example bdg_pairing
H4   = hopping(4; pbc=true)
lind = MajoranaLindblad(H4; loss_ops=[sqrt(0.3) * ComplexF64[1, 0, 0, 0]],
                        dephasing_ops=[(ComplexF64[1, 1, 0, 0] / sqrt(2), 0.5)])

maj = MajoranaState(CorrelationState(SlaterState(L=4, N=2, config="Z2")))
evolve!(maj, lind, 1.0)
(density        = round.(density(maj); digits=3),
 anomalous_norm = round(norm(anomalous_correlation(maj)); digits=4))
```

For a topological p-wave Hamiltonian and its Majorana edge modes, see the
[Kitaev Chain](kitaev-chain.md) example.
