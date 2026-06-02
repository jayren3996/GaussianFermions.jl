include("../src/GaussianFermions.jl")
using LinearAlgebra, Main.GaussianFermions

#--------------------------------------------------------------------------------
# Measurement-induced behaviour: steady-state mutual information of two antipodal
# regions of a monitored hopping ring, as a function of monitoring strength γ.
# (Illustrative / small sizes — not a production finite-size-scaling run.)
#--------------------------------------------------------------------------------

"""
Nodal monitoring modes dᵢ⁺ = (cᵢ₋₁⁺ + cᵢ⁺ + cᵢ₊₁⁺)/√3, one per site (PBC).
"""
function nodal_monitors(L; γ)
    V = ComplexF64[1, 1, 1] / sqrt(3)
    [OccupationMonitor(QuasiMode([mod(i-2, L)+1, i, mod(i, L)+1], V, L); γ=γ) for i in 1:L]
end

"""
Average mutual information I(A:B) over a monitored trajectory ensemble, where A
and B are two opposite quarter-rings.
"""
function mi_vs_gamma(; L=32, γ=1.0, dt=0.05, tspan=20.0, ntraj=32)
    H = hopping(L; pbc=true)
    A = 1:(L ÷ 4)
    B = (L ÷ 2 + 1):(3L ÷ 4)
    res = ensemble(() -> SlaterState(L=L, N=L ÷ 2, config="Z2"),
                   H, nodal_monitors(L; γ);
                   ntraj=ntraj, tspan=tspan, dt=dt,
                   observables=(I = s -> mutual_information(s, collect(A), collect(B)),))
    # late-time average (last 20% of the trajectory)
    tail = res.mean.I[(end - length(res.saveat) ÷ 5 + 1):end]
    sum(tail) / length(tail)
end

for γ in (0.2, 0.5, 1.0, 2.0)
    I = mi_vs_gamma(; L=32, γ=γ, ntraj=24, tspan=15.0)
    println("γ = $γ\t⟨I(A:B)⟩ ≈ $(round(I; digits=4))")
end
