using GaussianFermions
using LinearAlgebra

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
# Stochastic trajectory methods (number-conserving API)
#--------------------------------------------------------------------------------
"Quantum-jump (MCWF) trajectory average of the site density."
function jump_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1])
    H = hopping(L; pbc=false)
    chans = monitor_channels(d, L; γ=γ)
    res = ensemble(() -> SlaterState(L=L, N=L ÷ 2, config="left"), H, chans;
                   ntraj=ntraj, tspan=T, dt=dt, observables=(n=density,))
    reduce(hcat, res.mean.n)            # L × nsteps
end

"Quantum-state-diffusion trajectory average of the site density."
function qsd_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1])
    U = propagator(hopping(L; pbc=false), dt)
    chans = monitor_channels(d, L; γ=γ)
    N = round(Int, T / dt)
    acc = zeros(L, N)
    for _ in 1:ntraj
        s = SlaterState(L=L, N=L ÷ 2, config="left")
        for k in 1:N
            evolve!(s, U)
            step_diffusive!(s, chans, dt)
            acc[:, k] .+= density(s)
        end
    end
    acc ./ ntraj
end

#--------------------------------------------------------------------------------
# Deterministic reference: correlation-matrix Lindblad master equation
#--------------------------------------------------------------------------------
"Exact density evolution from the deterministic `CorrelationLindblad` generator."
function lindblad_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, d=[1, 3, 1])
    H = hopping(L; pbc=false)
    lind = CorrelationLindblad(H, monitor_channels(d, L; γ=γ))
    state = CorrelationState(SlaterState(L=L, N=L ÷ 2, config="left"))
    t = dt:dt:T
    reduce(hcat, [(evolve!(state, lind, dt); density(state)) for _ in t])
end

#--------------------------------------------------------------------------------
# trajectory vs deterministic Lindblad: agreement improves as 1/√ntraj
function main(; L=32, d=[1, 3, 1], ntraj=1000, T=10.0)
    den_ref = lindblad_den_evo(; L=L, d=d, T=T)
    den_jmp = jump_den_evo(; L=L, d=d, ntraj=ntraj, T=T)
    den_qsd = qsd_den_evo(; L=L, d=d, ntraj=ntraj, T=T)
    println("Jump vs Lindblad:  ", norm(den_jmp - den_ref))
    println("QSD  vs Lindblad:  ", norm(den_qsd - den_ref))
    den_ref, den_jmp, den_qsd
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
