# Trajectories Vs Lindblad

The deterministic Lindblad equation and trajectory unravelings describe the same
average dynamics when the channel conventions match. GaussianFermions.jl leaves the
sampling loop to the caller.

```julia
using GaussianFermions, Random

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

L = 16
dt = 0.05
rng = Xoshiro(1)
U = propagator(hopping(L; pbc=false), dt)
channels = [dephasing(i, L; γ=0.5) for i in 1:L]

traj = SlaterState(L=L, N=L ÷ 2, config="left")
for _ in 1:100
    mcwf_step!(traj, U, channels, dt; rng)
end
```

The deterministic reference uses the same channel objects:

```julia
lind = CorrelationLindblad(hopping(L; pbc=false), channels)
state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
for _ in 1:100
    evolve!(state, lind, dt)
end
```

See `example/Dephase.jl` for a full comparison between MCWF, QSD, Dirac Lindblad,
and Majorana Lindblad density evolution.
