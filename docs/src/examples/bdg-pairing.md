# BdG Pairing

Use `MajoranaState` and `BdGHamiltonian` when the Hamiltonian or bath can generate
anomalous correlations.

```@example bdg_pairing
using GaussianFermions, LinearAlgebra

A = zeros(ComplexF64, 4, 4)
B = zeros(ComplexF64, 4, 4)
B[1, 2] = 0.35
B[2, 1] = -0.35
B[3, 4] = -0.2
B[4, 3] = 0.2

H = BdGHamiltonian(A, B)
s = MajoranaState([0, 0, 0, 0])

evolve!(s, H, 0.75)
(density=round.(density(s); digits=3),
 anomalous_norm=round(norm(anomalous_correlation(s)); digits=4))
```

Ground and thermal BdG states use the same Hamiltonian:

```@example bdg_pairing
eps = quasiparticle_energies(H)
gs = groundstate(H)
(minimum_energy=round(minimum(eps); digits=4),
 ground_particle_number=round(particle_number(gs); digits=4))
```
