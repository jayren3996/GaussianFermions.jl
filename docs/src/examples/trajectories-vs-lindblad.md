# Trajectories vs Lindblad

The deterministic Lindblad equation and its stochastic unravelings describe the same
average dynamics of **linear** observables when the channel conventions match.
GaussianFermions.jl leaves the sampling loop to the caller; this page shows the site
density `n(t)` agreeing across four engines on a hopping chain with occupation
monitoring `D[√γ nᵢ]`.

## The MCWF step

```julia
using GaussianFermions, LinearAlgebra, Random

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

## Deterministic references

The Dirac (`CorrelationLindblad`) and Majorana (`MajoranaLindblad`) master equations
agree with each other to machine precision on number-conserving channels:

```@example tvl
using GaussianFermions, LinearAlgebra

L = 12; γ = 0.5; dt = 0.1; T = 4.0
chans = [dephasing(i, L; γ=γ) for i in 1:L]

dirac = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
lindD = CorrelationLindblad(hopping(L; pbc=false), chans)

majo  = MajoranaState(CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left")))
lindM = MajoranaLindblad(hopping(L; pbc=false), chans)

for _ in 1:round(Int, T / dt)
    evolve!(dirac, lindD, dt)
    evolve!(majo,  lindM, dt)
end
round(norm(density(dirac) - density(majo)); digits=12)
```

## Trajectory averages converge to the master equation

A Monte-Carlo wave-function average of the same density approaches the deterministic
result as `1/√ntraj` (stochastic — fixed seed for reproducibility):

```julia
function mcwf_density(; L=12, γ=0.5, dt=0.1, T=4.0, ntraj=400, rng=Xoshiro(1))
    U = propagator(hopping(L; pbc=false), dt)
    chans = [dephasing(i, L; γ=γ) for i in 1:L]
    acc = zeros(L)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for _ in 1:round(Int, T / dt); mcwf_step!(s, U, chans, dt; rng); end
        acc .+= density(s)
    end
    acc ./ ntraj
end

# ‖mcwf_density() − density(dirac)‖ ≈ 0.05  (≈ 1/√ntraj sampling error)
```

`example/Dephase.jl` runs all four engines — MCWF, quantum-state diffusion (QSD),
Dirac Lindblad, and Majorana Lindblad — and prints the pairwise differences. The
[Monitoring Protocols](monitoring-protocols.md) example contrasts the same
unravelings on a **nonlinear** observable, where they no longer coincide.
