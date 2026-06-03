# Free-Fermion Chain

This workflow shows the basic number-conserving path: prepare a Slater determinant,
evolve under a hopping Hamiltonian, and measure density and entanglement.

```@example freefermion
using GaussianFermions

L = 12
H = hopping(L; pbc=true)
s = SlaterState(L=L, N=L ÷ 2, config="Z2")

S0 = entanglement_entropy(s, 1:(L ÷ 2))
evolve!(s, H, 1.0)
S1 = entanglement_entropy(s, 1:(L ÷ 2))

(initial_entropy=round(S0; digits=4),
 final_entropy=round(S1; digits=4),
 particle_number=particle_number(s))
```

For fixed time steps, precompute the propagator:

```julia
U = propagator(H, 0.05)
for _ in 1:100
    evolve!(s, U)
end
```

The full script `example/FreeFermion.jl` extends this into a monitored trajectory
average.
