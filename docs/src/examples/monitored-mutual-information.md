# Monitored Mutual Information

Measurement-induced calculations usually require a caller-owned loop: step a
trajectory, compute an observable, and average over trajectories or over a late-time
window.

```julia
using GaussianFermions, Random

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
            apply_click!(ch, s, work)
            normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end

L = 32
A = collect(1:(L ÷ 4))
B = collect((L ÷ 2 + 1):(3L ÷ 4))
U = propagator(hopping(L; pbc=true), 0.05)
channels = nodal_monitors(L; γ=1.0)
rng = Xoshiro(1)

s = SlaterState(L=L, N=L ÷ 2, config="Z2")
for _ in 1:200
    mcwf_step!(s, U, channels, 0.05; rng)
end
mutual_information(s, A, B)
```

The full script `example/MI.jl` sweeps the monitoring strength and averages the
late-time mutual information of two antipodal regions.
