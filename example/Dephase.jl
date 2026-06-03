using GaussianFermions
using LinearAlgebra
using Random

#--------------------------------------------------------------------------------
# Dephasing model: cross-check the stochastic trajectory engines against the
# deterministic correlation-matrix Lindblad master equation.
#
#   H   = Σ cᵢ⁺cᵢ₊₁ + h.c.                  (hopping chain)
#   Lᵢ  = √γ dᵢ⁺dᵢ ,  dᵢ⁺ = Σₐ dₐ cᵢ₊ₐ⁺      (occupation monitoring of nodal modes)
#
# Quantum-jump (MCWF), quantum-state-diffusion (QSD), and the deterministic
# `CorrelationLindblad` master equation must all give the same averaged site
# density n(t); the trajectory error shrinks as 1/√ntraj.
#
# (The Majorana/BdG covariance-matrix Lindblad cross-check from the legacy
#  `Quadratic.jl` API is deferred until the Stage 2 `MajoranaLindblad` slice.)
#--------------------------------------------------------------------------------

"Monitoring modes dᵢ⁺ = Σₐ dₐ cᵢ₊ₐ⁺ (one block of `length(d)` sites per mode)."
function monitor_modes(d, L)
    nd = normalize(ComplexF64.(d))
    num = length(d)
    [QuasiMode(i:i+num-1, nd, L) for i in 1:num:L-num+1]
end

"Occupation-monitor channels for the nodal modes."
monitor_channels(d, L; γ) = [OccupationMonitor(qm; γ=γ) for qm in monitor_modes(d, L)]

#--------------------------------------------------------------------------------
# Stochastic trajectory methods (number-conserving API) — the trajectory steps and
# averaging loops live here; the library supplies only the per-step primitives.
#--------------------------------------------------------------------------------
# One MCWF (quantum-jump) step: unitary, then per-channel click / no-click.
function mcwf_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        rate, work = jump_rate(ch, s, dt)
        if rand(rng) < rate
            apply_click!(ch, s, work); normalize!(s)
        else
            apply_noclick!(ch, s, dt)
        end
    end
    s
end

# One QSD (diffusive) step: unitary, then a weak occupation filter exp(α nₘ) per mode.
function qsd_step!(s, U, channels, dt; rng=Random.default_rng())
    evolve!(s, U)
    for ch in channels
        qm = ch.mode
        α = randn(rng) * sqrt(ch.γ * dt) + (2 * density(s, qm) - 1) * ch.γ * dt
        weak_measure!(s, qm, α)
    end
    s
end

"Quantum-jump (MCWF) trajectory average of the site density."
function jump_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1], rng=Random.default_rng())
    U = propagator(hopping(L; pbc=false), dt)
    chans = monitor_channels(d, L; γ=γ)
    N = round(Int, T / dt)
    acc = zeros(L, N)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for k in 1:N
            mcwf_step!(s, U, chans, dt; rng)
            acc[:, k] .+= density(s)
        end
    end
    acc ./ ntraj
end

"Quantum-state-diffusion trajectory average of the site density."
function qsd_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1], rng=Random.default_rng())
    U = propagator(hopping(L; pbc=false), dt)
    chans = monitor_channels(d, L; γ=γ)
    N = round(Int, T / dt)
    acc = zeros(L, N)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for k in 1:N
            qsd_step!(s, U, chans, dt; rng)
            acc[:, k] .+= density(s)
        end
    end
    acc ./ ntraj
end

#--------------------------------------------------------------------------------
# Deterministic reference: correlation-matrix Lindblad master equation
#--------------------------------------------------------------------------------
"Exact density evolution from the deterministic Dirac `CorrelationLindblad` generator."
function lindblad_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, d=[1, 3, 1])
    H = hopping(L; pbc=false)
    lind = CorrelationLindblad(H, monitor_channels(d, L; γ=γ))
    state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
    t = dt:dt:T
    reduce(hcat, [(evolve!(state, lind, dt); density(state)) for _ in t])
end

"Exact density evolution from the deterministic Majorana `MajoranaLindblad` generator."
function majorana_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, d=[1, 3, 1])
    H = hopping(L; pbc=false)
    lind = MajoranaLindblad(H, monitor_channels(d, L; γ=γ))
    state = MajoranaState(CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left")))
    t = dt:dt:T
    reduce(hcat, [(evolve!(state, lind, dt); density(state)) for _ in t])
end

#--------------------------------------------------------------------------------
# All four methods must agree; the Dirac and Majorana master equations agree to
# machine precision, and the trajectory averages approach them as 1/√ntraj.
function main(; L=32, d=[1, 3, 1], ntraj=1000, T=10.0)
    den_ref = lindblad_den_evo(; L=L, d=d, T=T)
    den_maj = majorana_den_evo(; L=L, d=d, T=T)
    den_jmp = jump_den_evo(; L=L, d=d, ntraj=ntraj, T=T)
    den_qsd = qsd_den_evo(; L=L, d=d, ntraj=ntraj, T=T)
    println("Majorana vs Dirac Lindblad: ", norm(den_maj - den_ref))
    println("Jump     vs Lindblad:       ", norm(den_jmp - den_ref))
    println("QSD      vs Lindblad:       ", norm(den_qsd - den_ref))
    den_ref, den_maj, den_jmp, den_qsd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
