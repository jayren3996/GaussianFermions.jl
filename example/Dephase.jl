include("../src/GaussianFermions.jl")
using LinearAlgebra, Main.GaussianFermions, Plots
#--------------------------------------------------------------------------------
# Dephasing model
#--------------------------------------------------------------------------------
"""
Hopping Hamiltonian:
    H = ∑ cᵢ⁺cᵢ₊₁ + cᵢ₊₁⁺cᵢ.
"""
function H_evo(;L, dt, PBC)
    H = diagm(1 => ones(L-1), -1 => ones(L-1))
    PBC && (H[L,1] = H[1,L] = 1)
    evo_operator(Hermitian(H), dt)
end
#--------------------------------------------------------------------------------
"""
Jump operator
    Lᵢ = ∑ₐ dₐ cᵢ₊ₐ 
"""
function qparticles(d, L)
    nd = normalize(d)
    num = length(d) 
    [QuasiMode(i:i+num-1, nd, L) for i in 1:num:L-num+1]
end


#--------------------------------------------------------------------------------
# Quantum trajectory
#--------------------------------------------------------------------------------
"""
Using quantum jump method.
"""
function jump_den_evo(;
    L=32, γ=0.5, dt=0.05, T=20, 
    times=64, PBC=false, 
    d=[1,1,1]
)
    Ut = H_evo(;L, dt, PBC)
    ls = QJParticle.(qparticles(d, L), γ * dt)
    s0 = FreeFermionState(L=L, N=L÷2, config="left")
    N = round(Int, T/dt)
    out = map(1:times) do _
        s = deepcopy(s0)
        den = zeros(L, N)
        for k = 1:N 
            apply!(Ut, s)
            apply!(ls, s)
            den[:, k] = diag(s)
        end
        den
    end
    sum(out) / times
end
#--------------------------------------------------------------------------------
"""
using quantum state diffusion.
"""
function qsd_den_evo(;
    L=32, γ=0.5, dt=0.05, T=20, 
    times=64, PBC=false,
    d=[1,1,1], threads=false
)
    Ut = H_evo(;L, dt, PBC)
    qms = qparticles(d, L)
    s0 = FreeFermionState(L=L, N=L÷2, config="left")
    N = round(Int, T/dt)
    out = map(1:times) do _
        s = deepcopy(s0)
        den = zeros(L, N)
        for k = 1:N 
            apply!(Ut, s)
            wiener!(qms, s, γ*dt; threads)
            den[:, k] = diag(s)
        end
        den
    end
    sum(out) / times
end

#--------------------------------------------------------------------------------
# Lindbladian
#--------------------------------------------------------------------------------
function majorana_den_evo(;
    L=32, γ=0.5, dt=0.05, T=20, PBC=false, d=[1,1,1]
)
    lind = begin
        H = diagm(1 => ones(L-1), -1 => ones(L-1))
        PBC && (H[L,1] = H[1,L] = 1)
        nd = normalize(d)
        num = length(d)
        mat = sqrt(γ) * nd * nd' 
        I = [i:i+num-1 for i in 1:num:L-num+1]
        M = fill(mat, length(I))
        quadraticlindblad_from_fermion(;H, M, I)
    end
    Γ = covariancematrix([ones(Int,L÷2); zeros(Int,L÷2)])
    t = dt:dt:T
    sol = lindblad_evo(lind, Γ, t)
    den = zeros(L,length(t))
    for i in eachindex(t)
        C = fermioncorrelation(sol[i], 1)
        den[:, i] = real.(diag(C))
    end
    den
end

#--------------------------------------------------------------------------------
function fermion_den_evo(;
    L=32, γ=0.5, dt=0.05, T=20, PBC=false, d=[1,1,1]
)
    lind = begin
        H = diagm(1 => ones(L-1), -1 => ones(L-1))
        PBC && (H[L,1] = H[1,L] = 1)
        nd = normalize(d)
        num = length(d)
        mat = sqrt(γ) * nd * nd' 
        I = [i:i+num-1 for i in 1:num:L-num+1]
        M = fill(mat, length(I))
        fermionlindblad(H, M, I)
    end

    Γ = diagm([ones(ComplexF64,L÷2); zeros(ComplexF64,L÷2)])
    t = dt:dt:T
    sol = lindblad_evo(lind, Γ, t)
    den = zeros(L,length(t))
    for i in eachindex(t)
        den[:, i] = real.(diag(sol[i]))
    end
    den
end

function main(;d=[1,3,1],times=1000)
    # ~ 2.45 s
    @time den1 = majorana_den_evo(;d)
    # ~ 1.59 s
    @time den2 = fermion_den_evo(;d)
    println("Majorana vs Fermion: $(norm(den1-den2))")

    # ~ 22 s
    @time den3 = jump_den_evo(;d,times)
    println("Jump vs Fermion: $(norm(den3-den2))")

    # ~ 21 s
    @time den4 = qsd_den_evo(;d,times)
    println("QSD vs Fermion: $(norm(den4-den2))")

    den1, den2, den3, den4 
end

res = main();
