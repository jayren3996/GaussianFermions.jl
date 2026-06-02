include("../src/GaussianFermions.jl")
using LinearAlgebra, SparseArrays, Main.GaussianFermions

#--------------------------------------------------------------------------------
# Dephasing model: cross-check the (new) number-conserving trajectory engine
# against the (stage-2) covariance-matrix Lindblad master equation.
#
#   H   = Σ cᵢ⁺cᵢ₊₁ + h.c.            (hopping chain)
#   Lᵢ  = √γ dᵢ⁺dᵢ ,  dᵢ⁺ = Σₐ dₐ cᵢ₊ₐ⁺   (occupation monitoring of nodal modes)
#
# All four methods (quantum jump, QSD, Majorana Lindblad, Dirac Lindblad) must
# give the same averaged site density n(t).
#--------------------------------------------------------------------------------

"Monitoring modes dᵢ⁺ = Σₐ dₐ cᵢ₊ₐ⁺ (one block of `length(d)` sites per mode)."
function monitor_modes(d, L)
    nd = normalize(ComplexF64.(d))
    num = length(d)
    [QuasiMode(i:i+num-1, nd, L) for i in 1:num:L-num+1]
end

#--------------------------------------------------------------------------------
# Trajectory methods (new number-conserving API)
#--------------------------------------------------------------------------------
"Quantum-jump (MCWF) trajectory average of the site density."
function jump_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1])
    H = hopping(L; pbc=false)
    chans = [OccupationMonitor(qm; γ=γ) for qm in monitor_modes(d, L)]
    res = ensemble(() -> SlaterState(L=L, N=L ÷ 2, config="left"), H, chans;
                   ntraj=ntraj, tspan=T, dt=dt, observables=(n=density,))
    reduce(hcat, res.mean.n)            # L × nsteps
end

"Quantum-state-diffusion trajectory average of the site density."
function qsd_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, ntraj=300, d=[1, 3, 1])
    U = propagator(hopping(L; pbc=false), dt)
    chans = [OccupationMonitor(qm; γ=γ) for qm in monitor_modes(d, L)]
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
# Deterministic Lindblad (covariance / correlation matrix — Quadratic.jl, unchanged)
#--------------------------------------------------------------------------------
function _lind_builder(L, γ, d)
    H = diagm(1 => ones(L - 1), -1 => ones(L - 1))
    nd = normalize(d)
    num = length(d)
    mat = sqrt(γ) * nd * nd'
    I_ = [i:i+num-1 for i in 1:num:L-num+1]
    M = fill(mat, length(I_))
    H, M, I_
end

function majorana_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, d=[1, 3, 1])
    H, M, I_ = _lind_builder(L, γ, d)
    lind = quadraticlindblad_from_fermion(; H=H, M=M, I=I_)
    Γ = covariancematrix([ones(Int, L ÷ 2); zeros(Int, L ÷ 2)])
    t = dt:dt:T
    sol = lindblad_evo(lind, Γ, collect(t))
    reduce(hcat, [real.(diag(fermioncorrelation(sol[i], 1))) for i in eachindex(t)])
end

function fermion_den_evo(; L=32, γ=0.5, dt=0.05, T=10.0, d=[1, 3, 1])
    H, M, I_ = _lind_builder(L, γ, d)
    lind = fermionlindblad(H, M, I_)
    Γ = diagm([ones(ComplexF64, L ÷ 2); zeros(ComplexF64, L ÷ 2)])
    t = dt:dt:T
    sol = lindblad_evo(lind, Γ, collect(t))
    reduce(hcat, [real.(diag(sol[i])) for i in eachindex(t)])
end

#--------------------------------------------------------------------------------
# trajectory vs Lindblad agreement improves as 1/√ntraj
function main(; d=[1, 3, 1], ntraj=1000)
    den_maj = majorana_den_evo(; d)
    den_fer = fermion_den_evo(; d)
    println("Majorana vs Dirac Lindblad: ", norm(den_maj - den_fer))
    den_jmp = jump_den_evo(; d, ntraj)
    println("Jump   vs Lindblad:         ", norm(den_jmp - den_fer))
    den_qsd = qsd_den_evo(; d, ntraj)
    println("QSD    vs Lindblad:         ", norm(den_qsd - den_fer))
    den_maj, den_fer, den_jmp, den_qsd
end

main();
