#---------------------------------------------------------------------------------------------------
# BdG / Majorana quadratic Hamiltonians
#---------------------------------------------------------------------------------------------------
export BdGHamiltonian

mutable struct BdGHamiltonian{T<:Real} <: AbstractQuadraticHamiltonian
    K::Matrix{T}
    cache::Union{Nothing,Tuple{Float64,Matrix{T}}}
end

function _check_square_even_matrix(M::AbstractMatrix, label::String)
    size(M, 1) == size(M, 2) ||
        throw(ArgumentError("$label must be square, got $(size(M))"))
    iseven(size(M, 1)) ||
        throw(ArgumentError("$label size must be even, got $(size(M, 1))"))
end

function BdGHamiltonian(K::AbstractMatrix; check::Bool=true, atol::Real=1e-10)
    _check_square_even_matrix(K, "BdG Majorana generator")
    Kmat = Matrix{Float64}(K)
    all(isfinite, Kmat) ||
        throw(ArgumentError("BdG Majorana generator entries must be finite"))
    if check
        isapprox(Kmat, -transpose(Kmat); atol) ||
            throw(ArgumentError("BdG Majorana generator must be antisymmetric"))
    end
    BdGHamiltonian{Float64}((Kmat - transpose(Kmat)) / 2, nothing)
end

function _majorana_hamiltonian_matrix(A::AbstractMatrix, B::AbstractMatrix)
    size(A) == size(B) ||
        throw(ArgumentError("BdG blocks A and B must have the same size, got $(size(A)) and $(size(B))"))
    size(A, 1) == size(A, 2) ||
        throw(ArgumentError("BdG block A must be square, got $(size(A))"))
    isapprox(A, A'; atol=1e-10) ||
        throw(ArgumentError("BdG block A must be Hermitian"))
    isapprox(B, -transpose(B); atol=1e-10) ||
        throw(ArgumentError("BdG pairing block B must be antisymmetric"))
    AR, AI = real.(A), imag.(A)
    BR, BI = real.(B), imag.(B)
    Matrix{Float64}([-AI - BI AR - BR; -AR - BR -AI + BI])
end

# Covariance generator K such that Γ(dt) = exp(K dt) Γ exp(K dt)ᵀ. The sign is fixed by
# requiring agreement with the package's number-conserving evolution C → conj(U) C Uᵀ,
# U = exp(-i h dt); equivalently K = +Hmaj in the legacy `majoranaform` algebra.
BdGHamiltonian(A::AbstractMatrix, B::AbstractMatrix; check::Bool=true, atol::Real=1e-10) =
    BdGHamiltonian(_majorana_hamiltonian_matrix(A, B); check, atol)

BdGHamiltonian(H::QuadraticHamiltonian) =
    BdGHamiltonian(_majorana_hamiltonian_matrix(Matrix(H.h), zeros(ComplexF64, nmodes(H), nmodes(H))); check=false)

nmodes(H::BdGHamiltonian) = size(H.K, 1) ÷ 2
Base.Matrix(H::BdGHamiltonian) = copy(H.K)

function propagator(H::BdGHamiltonian, dt::Real)
    if H.cache !== nothing && H.cache[1] == Float64(dt)
        return H.cache[2]
    end
    O = exp(Float64(dt) .* H.K)
    H.cache = (Float64(dt), O)
    O
end

function evolve!(s::MajoranaState, O::AbstractMatrix)
    size(O) == size(s.Gamma) ||
        throw(ArgumentError("Majorana propagator size $(size(O)) does not match covariance size $(size(s.Gamma))"))
    Omat = Matrix{Float64}(O)
    G = Omat * s.Gamma * transpose(Omat)
    s.Gamma = (G - transpose(G)) / 2
    s
end

evolve!(s::MajoranaState, H::BdGHamiltonian, dt::Real) = evolve!(s, propagator(H, dt))
evolve(s::MajoranaState, args...) = evolve!(copy(s), args...)
