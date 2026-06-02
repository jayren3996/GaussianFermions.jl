include("../src/GaussianFermions.jl")
using DelimitedFiles, LinearAlgebra, Main.GaussianFermions
#--------------------------------------------------------------------------------
"""
H = ∑ cᵢ⁺cᵢ₊₁ + cᵢ₊₁⁺cᵢ.
"""
function hopping_ham(L::Integer; PBC::Bool=false)
    H = diagm(1 => ones(L-1), 1 => ones(L-1))
    PBC && (H[L,1] = H[1,L] = 1)
    H
end
#--------------------------------------------------------------------------------
"""
Jump operator:
    Lᵢ = √γ * dᵢ⁺dᵢ,
    dᵢ⁺ = (cᵢ₋₁⁺ + α cᵢ⁺ + cᵢ₊₁⁺) / √2.
"""
function nodal_jump(L::Integer; a=1.0)
    V = [1 , a, 1] / sqrt(2+abs2(a))
    l1 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 1:3:L-2]
    l2 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 2:3:L-1]
    l3 = [QuasiMode([i, mod(i,L)+1, mod(i+1,L)+1], V, L) for i in 3:3:L]
    l1, l2, l3
end

#--------------------------------------------------------------------------------
# Main
#--------------------------------------------------------------------------------
function ent_evo(;L=128, γ=0.5, dt=0.05, T=500, a=1, config="Z2", p::Integer=10)
    N = round(Int, T/dt)
    evo = evo_operator(Hermitian(hopping_ham(L, PBC=true)), dt)
    l1, l2, l3 = nodal_jump(L; a)
    s = FreeFermionState(L=L, N=L÷2, config=config)
    EE = zeros(N÷p)
    hL, oL = L÷2, L÷8
    for k = 1:N
        s = evo * s
        wiener!(l1, s, γ*dt)
        wiener!(l2, s, γ*dt)
        wiener!(l3, s, γ*dt)
        if iszero(mod(k,p))
            ee = zeros(hL)
            Threads.@threads for i in 1:hL 
                ind1 = mod.(i-1:oL+i-2, L) .+ 1
                ind2 = mod.(hL+i-1:hL+oL+i-2, L) .+ 1
                ee[i] = ent_S(s, ind1, ind2)
            end
            EE[k÷p] = sum(ee) / hL
            writedlm("$L-$γ.dat", EE)
            GC.gc()
            iszero(mod(k,200)) && (println("L=$L, γ=$γ, t=$(k*dt)"))
        end
    end
end

gs = 0.05:0.05:2
for g in gs
    ent_evo(L=128, γ=g)
    ent_evo(L=256, γ=g)
    ent_evo(L=512, γ=g)
end

gs = 2.2:0.2:5
for g in gs
    ent_evo(L=128, γ=g, dt=0.02, p=25)
    ent_evo(L=256, γ=g, dt=0.02, p=25)
    ent_evo(L=512, γ=g, dt=0.02, p=25)
end

gs = 5.5:0.5:10
for g in gs
    ent_evo(L=128, γ=g, dt=0.01, p=50)
    ent_evo(L=256, γ=g, dt=0.01, p=50)
    ent_evo(L=512, γ=g, dt=0.01, p=50)
end

gs = 12:2:20
for g in gs
    ent_evo(L=128, γ=g, dt=0.005, p=100)
    ent_evo(L=256, γ=g, dt=0.005, p=100)
    ent_evo(L=512, γ=g, dt=0.005, p=100)
end
