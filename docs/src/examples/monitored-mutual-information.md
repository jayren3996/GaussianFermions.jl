# Mutual & Tripartite Information

Measurement-induced calculations average a nonlinear diagnostic over a monitored
ensemble or a late-time window. The mutual information `I(A:B)` and the tripartite
information `I₃(A:B:C)` of separated regions are the standard order parameters: as the
monitoring rate `γ` grows, correlations between distant regions are suppressed and
`I₃` collapses toward zero.

## Nodal occupation monitoring

```julia
using GaussianFermions, LinearAlgebra, Random

# Nodal monitoring modes dᵢ⁺ = (cᵢ₋₁⁺ + cᵢ⁺ + cᵢ₊₁⁺)/√3, one per site (PBC).
function nodal_monitors(L; γ)
    V = ComplexF64[1, 1, 1] / sqrt(3)
    [OccupationMonitor(QuasiMode([mod(i-2, L)+1, i, mod(i, L)+1], V, L); γ=γ)
     for i in 1:L]
end

function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        p, work = jump_rate(ch, s, dt)
        if rand(rng) < p
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end
```

## Late-time mutual and tripartite information

`A`, `B`, `C` are three consecutive quarter-rings; the diagnostics are averaged over
the last quarter of each trajectory and over the ensemble (stochastic — fixed seed):

```julia
function info_vs_gamma(; L=16, γ=1.0, dt=0.1, tspan=10.0, ntraj=24, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=true), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    A = collect(1:(L ÷ 4)); B = collect((L ÷ 4 + 1):(L ÷ 2))
    C = collect((L ÷ 2 + 1):(3L ÷ 4))
    nsteps = round(Int, tspan / dt); tail = nsteps ÷ 4
    ΣI = 0.0; ΣI3 = 0.0; nsamp = 0
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="Z2")
        for k in 1:nsteps
            mcwf_step!(s, U, chans, dt; rng)
            if k > nsteps - tail
                ΣI  += mutual_information(s, A, B)
                ΣI3 += tripartite_information(s, A, B, C)
                nsamp += 1
            end
        end
    end
    (mutual=ΣI / nsamp, tripartite=ΣI3 / nsamp)
end

for γ in (0.2, 1.0, 3.0)
    r = info_vs_gamma(; γ=γ)
    println("γ = $γ  ⟨I(A:B)⟩ ≈ $(round(r.mutual; digits=3))",
            "   ⟨I₃⟩ ≈ $(round(r.tripartite; digits=3))")
end
```

Representative output (small ensemble; real finite-size scaling needs larger `L` and
`ntraj`):

```text
γ = 0.2  ⟨I(A:B)⟩ ≈ 1.270   ⟨I₃⟩ ≈ -0.213
γ = 1.0  ⟨I(A:B)⟩ ≈ 0.952   ⟨I₃⟩ ≈ -0.024
γ = 3.0  ⟨I(A:B)⟩ ≈ 0.365   ⟨I₃⟩ ≈  0.001
```

Both `I(A:B)` and `|I₃|` shrink as monitoring strengthens; `I₃ → 0` in the
strongly-monitored (area-law) regime. The full sweep is in
[`example/MI.jl`](https://github.com/jayren3996/GaussianFermions.jl/blob/main/example/MI.jl).
See [Measurement-Induced Transition](measurement-induced-transition.md) for the
entanglement-entropy view of the same physics.
