#---------------------------------------------------------------------------------------------------
# Covariance matrix
#---------------------------------------------------------------------------------------------------
export covariancematrix
"""
Create a covariance matrix from a fermion occupied state.
"""
function covariancematrix(n::AbstractVector{<:Integer})
    N = length(n)
    D = diagm(-2 * n .+ 1)
    Z = zeros(N, N)
    [Z D; -D Z]
end
#---------------------------------------------------------------------------------------------------
export majoranaform
"""
    majoranaform(A::AbstractMatrix, B::AbstractMatrix)

Conversion between fermion and Majorana representation.

Fermion quadratic form: 
    Ĥ = 1/2 ∑(Aᵢⱼ cᵢ⁺cⱼ + Bᵢⱼcᵢ⁺cⱼ⁺ + h.c.)
Majorana quadratic form:
    Ĥ = -i/4 ∑ Hᵢⱼ ωᵢωⱼ
"""
function majoranaform(A::AbstractMatrix, B::AbstractMatrix)
    AR, AI, BR, BI = real(A), imag(A), real(B), imag(B)
    [-AI-BI AR-BR; -AR-BR -AI+BI]
end
#---------------------------------------------------------------------------------------------------
"""
Return the Majorana quadratic form from U(1)-conserving free fermion form:
    ∑Aᵢⱼ cᵢ⁺cⱼ ⟹ -i/4 ∑ Hᵢⱼ ωᵢωⱼ
"""
function majoranaform(A::AbstractMatrix)
    AR, AI = real(A), imag(A)
    [-AI AR; -AR -AI]
end
#---------------------------------------------------------------------------------------------------
export fermioncorrelation
"""
Correlation:
    A = ⟨cᵢ⁺cⱼ⟩
    B = ⟨cᵢcⱼ⟩
"""
function fermioncorrelation(Γ::AbstractMatrix{<:Real})
    n = size(Γ, 1) ÷ 2
    Γ11, Γ12, Γ21, Γ22 = Γ[1:n,1:n], Γ[1:n,n+1:2n], Γ[n+1:2n,1:n], Γ[n+1:2n,n+1:2n]
    A = (Γ21 - Γ12 + 1im * Γ11 + 1im * Γ22) / 4 + I / 2
    B = (Γ21 + Γ12 + 1im * Γ11 - 1im * Γ22) / 4
    A, B
end
#---------------------------------------------------------------------------------------------------
"""
Correlation:
    1: ⟨cᵢ⁺cⱼ⟩
    2: ⟨cᵢcⱼ⟩
    3: ⟨cᵢ⁺cⱼ⁺⟩
"""
function fermioncorrelation(Γ::AbstractMatrix{<:Real}, i::Integer)
    n = size(Γ, 1) ÷ 2
    Γ11, Γ12, Γ21, Γ22 = Γ[1:n,1:n], Γ[1:n,n+1:2n], Γ[n+1:2n,1:n], Γ[n+1:2n,n+1:2n]
    if i == 1
        (Γ21 - Γ12 + 1im * Γ11 + 1im * Γ22) / 4 + I / 2
    elseif i == 2
        (Γ21 + Γ12 + 1im * Γ11 - 1im * Γ22) / 4
    elseif i == 3
        (-Γ21 - Γ12 + 1im * Γ11 - 1im * Γ22) / 4
    else
        error("i should equals to 1, 2, or 3.")
    end
end
#---------------------------------------------------------------------------------------------------
function majorana_eigen(H::AbstractMatrix{<:Real})
    n = size(H, 1) ÷ 2
    S, V, vals = schur(H)
    λ = abs.(imag(vals))
    O = Matrix{Float64}(undef, 2n, 2n)
    for i=1:n
        if S[2i-1, 2i] > 0
            O[:, i] = V[:, 2i-1]
            O[:, i+n] = V[:, 2i]
        else
            O[:, i] = V[:, 2i]
            O[:, i+n] = V[:, 2i-1]
        end
    end
    λ, O
end
#---------------------------------------------------------------------------------------------------
function majorana_eigvals(H::AbstractMatrix{<:Real})
    vals = eigvals(Hermitian(1im*H),sortby=imag)
    vals[end÷2+1:end]
end
#---------------------------------------------------------------------------------------------------
function vonneumann_entropy(vals::AbstractVector{<:Real})
    EE = 0.0
    for x in vals
        x > 1.0 + 1e-6 && error("Got an invalid value λ = $x.")
        a = (1 + x) / 2
        b = 1 - a
        (a < 1e-14 || b < 1e-14) && continue
        EE -= a * log(a) + b * log(b)
    end
    EE
end
#---------------------------------------------------------------------------------------------------
"""
Entropy of the density matrix
"""
function entropy(Γ::AbstractMatrix{<:Real})
    majorana_eigvals(Γ) |> vonneumann_entropy
end
#---------------------------------------------------------------------------------------------------
function entropy(Γ::AbstractMatrix{<:Real}, i::AbstractVector{<:Integer})
    n = size(Γ, 1) ÷ 2
    ind = [i; i .+ n]
    majorana_eigvals(Γ[ind, ind]) |> vonneumann_entropy
end
#---------------------------------------------------------------------------------------------------
function entropy(Γ::AbstractMatrix{<:Real}, i::AbstractVector{<:Integer}, j::AbstractVector{<:Integer})
    SA = entropy(Γ, i)
    SB = entropy(Γ, j)
    SAB = entropy(Γ, vcat(i,j))
    SA + SB - SAB
end



#---------------------------------------------------------------------------------------------------
# Majorana Lindblad with up to quadratic jumps
#---------------------------------------------------------------------------------------------------
export quadraticlindblad
"""
Quadratic Lindblad with free hamiltonian
    H = -i/4 ∑ⱼₖ Hⱼₖ ωⱼωₖ
and jump operators:
    L_r = ∑ⱼ Lʳⱼ ωⱼ
    L_s = -i/4 ∑ⱼ Mˢⱼₖ ωⱼωₖ
where Mˢ is real, anti-symmetric.

The evolution of the covariance (Γ = i⟨ωᵢωⱼ⟩-iδᵢⱼ) is
    ∂ₜΓ = Xᵀ⋅Γ + Γ⋅X - ∑ₛ Mˢ⋅Γ⋅Mˢ + Y
In most cases, Mˢ is local operator, so we store the local operators in `M` and positions in `I`.
"""
struct QuardraticLindblad{
    T1 <: AbstractMatrix{<:Real}, 
    T2 <: AbstractMatrix{<:Real}, 
    T3 <: AbstractMatrix{<:Real},
    T4 <: AbstractVector{<:Integer}
} 
    X::T1
    Y::T2
    M::Vector{T3}
    I::Vector{T4}
end
#---------------------------------------------------------------------------------------------------
"""
The evolution of the covariance (Γ = i⟨ωᵢωⱼ⟩-iδᵢⱼ) is
    ∂ₜΓ = Xᵀ⋅Γ + Γ⋅X - ∑ₛ Mˢ⋅Γ⋅Mˢ + Y
where
    B  = ∑ᵣ Lʳᵢ L̄ʳⱼ
    X  = H - 2 Re[B] + 1/2 ∑ₛ (Mˢ)^2
    Y  = 4 Im[B]
"""
function quadraticlindblad(
    H::AbstractMatrix, 
    L::AbstractMatrix, 
    M::AbstractVector{<:AbstractMatrix},
    I::AbstractVector{<:AbstractVector{<:Integer}}
)
    B = L * L' 
    X = H - 2 * real(B)
    for (i, Ms) in enumerate(M)
        ind = I[i]
        X[ind, ind] .+= Ms^2 / 2
    end
    Y = 4 * imag(B)
    QuardraticLindblad(X, Y, M, I)
end
#---------------------------------------------------------------------------------------------------
"""
Quasi-free Lindbladian
"""
function quadraticlindblad(
    H::AbstractMatrix, 
    L::AbstractMatrix
)
    B = L * L' 
    X = H - 2 * real(B)
    Y = 4 * imag(B)
    M = Matrix{Float64}[]
    I = Vector{Int64}[]
    QuardraticLindblad(X, Y, M, I)
end
#---------------------------------------------------------------------------------------------------
export quadraticlindblad_from_fermion
"""
From Dirac fermion form
"""
function quadraticlindblad_from_fermion(;H::AbstractMatrix, L=nothing, M=nothing, I=nothing)
    N = size(H, 1)
    Hm = majoranaform(H)
    Lm = isnothing(L) ? zeros(2N, 0) : begin
        c = L[1:N, :]
        d = L[N+1:2N, :]
        [(c+d)/2; 1im(d-c)/2]
    end
    Mm = isnothing(M) ? Matrix{Float64}[] : majoranaform.(M)
    Im = if isnothing(I)
        Vector{Int64}[]
    else
        inds = [mod.(i .- 1, N) .+ 1 for i in I] 
        [[ind; ind .+ N] for ind in inds]
    end
    quadraticlindblad(Hm, Lm, Mm, Im)
end

#---------------------------------------------------------------------------------------------------
# Evolution of Covariance Matrix
#---------------------------------------------------------------------------------------------------
"""
Tangent vector of the evolution of the covariance (Γ = i⟨ωᵢωⱼ⟩-iδᵢⱼ)
    ∂ₜΓ = Xᵀ⋅Γ + Γ⋅X - ∑ₛ Mˢ⋅Γ⋅Mˢ + Y
"""
function *(ql::QuardraticLindblad, Γ::AbstractMatrix{<:Real})
    out = Γ * ql.X
    out -= transpose(out)
    for (i, Ms) in enumerate(ql.M)
        ind = ql.I[i]
        out[ind, ind] .-= Ms * Γ[ind, ind] * Ms
    end
    out + ql.Y
end



#---------------------------------------------------------------------------------------------------
# Fermion Lindblad with quadratic jump
#---------------------------------------------------------------------------------------------------
struct FermionLindblad{T1 <: AbstractMatrix, T2 <: AbstractMatrix, T3 <: Integer} 
    X::T1
    Z::Vector{T2}
    I::Vector{Vector{T3}}
end
#---------------------------------------------------------------------------------------------------
export fermionlindblad
function fermionlindblad(
    H::AbstractMatrix, 
    M::AbstractVector{<:AbstractMatrix},
    I::AbstractVector{<:AbstractVector{<:Integer}}
)
    Z = conj.(M)
    X = -1im * conj(H) 
    I = [mod.(i .- 1, size(H,1)) .+ 1 for i in I]
    for (i, ind) in enumerate(I)
        X[ind, ind] .-= Z[i]^2 / 2
    end
    FermionLindblad(X, Z, I)
end
#---------------------------------------------------------------------------------------------------
function *(fl::FermionLindblad, G::AbstractMatrix)
    out = G * fl.X
    out = out + out'
    for (i, ind) in enumerate(fl.I)
        out[ind, ind] .+= fl.Z[i] * G[ind, ind] * fl.Z[i]
    end
    out
end

#---------------------------------------------------------------------------------------------------
# Differential equations
#---------------------------------------------------------------------------------------------------
export lindblad_evo
function lindblad_evo(
    lind::Union{<:QuardraticLindblad,<:FermionLindblad}, 
    Γ::Matrix, t::AbstractVector{<:Real};
    reltol=1e-8, abstol=1e-8
)
    tspan = (0.0, t[end])
    prob = ODEProblem((u, p, t) -> (lind*u), Γ, tspan)
    sol = solve(prob, Tsit5(); reltol, abstol,saveat=t)
    sol.u
end
