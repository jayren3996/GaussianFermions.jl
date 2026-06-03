# Kitaev Chain & Bogoliubov Spectrum

The Kitaev p-wave chain is the canonical topological BdG model. With `t = Δ` and open
boundaries, the bulk is gapped while two unpaired Majorana modes sit at the ends — an
exact zero-energy mode in the topological phase (`|μ| < 2t`) that disappears in the
trivial phase (`|μ| > 2t`).

In the `BdGHamiltonian(A, B)` convention, `A` carries the hopping `−t` and on-site
`−μ`, and `B` carries the p-wave pairing `Δ` (antisymmetric):

```@example kitaev
using GaussianFermions, LinearAlgebra

function kitaev_blocks(L; t=1.0, Δ=1.0, μ=0.0)
    A = zeros(ComplexF64, L, L)
    B = zeros(ComplexF64, L, L)
    for j in 1:L
        A[j, j] = -μ
    end
    for j in 1:L-1
        A[j, j+1] = -t; A[j+1, j] = -t      # hopping (Hermitian)
        B[j, j+1] =  Δ; B[j+1, j] = -Δ      # p-wave pairing (antisymmetric)
    end
    A, B
end
nothing # hide
```

## Topological vs trivial spectrum

```@example kitaev
Atop, Btop = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.0)   # topological
Atriv, Btriv = kitaev_blocks(12; t=1.0, Δ=1.0, μ=3.0) # trivial

εtop  = quasiparticle_energies(BdGHamiltonian(Atop, Btop))
εtriv = quasiparticle_energies(BdGHamiltonian(Atriv, Btriv))

(topological_lowest = round.(εtop[1:3]; digits=4),
 trivial_lowest     = round.(εtriv[1:3]; digits=4))
```

The topological spectrum has an exact zero mode (the Majorana edge pair) below a bulk
gap of `2t`; the trivial spectrum is fully gapped.

## Ground state and pairing correlations

!!! note "Zero-mode degeneracy"
    Exactly at `μ = 0` the ground state is parity-degenerate, so the Nambu occupation
    is ill-defined and `groundstate` raises an error. Evaluate it at a slightly
    detuned point (still topological), which selects a definite-parity ground state.

```@example kitaev
A, B = kitaev_blocks(12; t=1.0, Δ=1.0, μ=0.5)   # detuned, still topological
H = BdGHamiltonian(A, B)
gs = groundstate(H)

(particle_number = round(particle_number(gs); digits=3),
 anomalous_norm  = round(norm(anomalous_correlation(gs)); digits=3),
 edge_density    = round.(density(gs)[[1, 6, 12]]; digits=3))
```

The end sites carry slightly enhanced density relative to the bulk — a footprint of
the localized edge modes. A finite-temperature state interpolates toward the
maximally mixed state:

```@example kitaev
ρβ = thermalstate(H; β=5.0)
round(particle_number(ρβ); digits=3)
```

See [BdG Pairing](bdg-pairing.md) for general pairing dynamics and baths.
