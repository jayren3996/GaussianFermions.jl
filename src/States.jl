#---------------------------------------------------------------------------------------------------
# Number-conserving (U(1)) fermionic Gaussian states
#---------------------------------------------------------------------------------------------------
export NumberConservingGaussianState, SlaterState, CorrelationState
export correlation_matrix, correlation, nmodes, ispure
export thermalstate, maximally_mixed

"""
    NumberConservingGaussianState{T}

Abstract supertype for number-conserving (U(1)-symmetric) fermionic Gaussian
states on `L` single-particle modes. Such a state is fully characterised by the
single-particle correlation matrix

    Cᵢⱼ = ⟨c⁺ᵢ cⱼ⟩            (L×L, Hermitian, eigenvalues in [0,1]).

Pure states have `C` a projector (`C² = C`); mixed states are general.

Interface every subtype implements:
    correlation_matrix(s)     -> L×L Hermitian C
    nmodes(s)                 -> L
Optional fast paths a subtype may specialise:
    correlation_matrix(s, A)  -> restricted block C[A,A]
    correlation(s, i, j)      -> scalar Cᵢⱼ
"""
abstract type NumberConservingGaussianState{T<:Number} end

#---------------------------------------------------------------------------------------------------
# Pure state — orbital (Slater) form
#---------------------------------------------------------------------------------------------------
"""
    SlaterState{T} <: NumberConservingGaussianState{T}

Pure number-conserving Gaussian state (single Slater determinant), stored by its
`L×N` orthonormal orbital matrix `B`:

    |ψ⟩ = ∏ₖ b⁺ₖ |0⟩,   b⁺ₖ = ∑ᵢ c⁺ᵢ Bᵢₖ.

Columns of `B` are orthonormal (`B'B = I`); the correlation matrix is the rank-`N`
projector `C = conj(B) * transpose(B)`. This is the efficient representation for
trajectory simulation.
"""
mutable struct SlaterState{T<:Number} <: NumberConservingGaussianState{T}
    B::Matrix{T}
end

# from occupied positions
"""
    SlaterState(pos, L; dtype=ComplexF64)

Fock state with the sites in `pos` occupied on `L` modes.
"""
function SlaterState(pos::AbstractVector{<:Integer}, L::Integer; dtype::DataType=ComplexF64)
    B = zeros(dtype, L, length(pos))
    for (i, ind) in enumerate(pos)
        B[ind, i] = one(dtype)
    end
    SlaterState(B)
end

# from a Boolean occupation mask
"""
    SlaterState(occ::AbstractVector{Bool}; dtype=ComplexF64)

Fock state from a Boolean occupation vector.
"""
function SlaterState(occ::AbstractVector{Bool}; dtype::DataType=ComplexF64)
    pos = findall(occ)
    SlaterState(pos, length(occ); dtype)
end

# config presets
"""
    SlaterState(; L, N, config="Z2")

Product-state presets on `L` modes with `N` particles. `config` ∈
`"left"`, `"right"`, `"center"`, `"random"`, or `"Zn"` (e.g. `"Z2"`).
"""
function SlaterState(; L::Integer, N::Integer, config::String="Z2")
    pos = if config == "center"
        l = (L-N)÷2
        l+1:l+N
    elseif config == "left"
        1:N
    elseif config == "right"
        L-N+1:L
    elseif config == "random"
        shuffle(1:L)[1:N]
    elseif config[1] == 'Z'
        n = parse(Int, config[2:end])
        1:n:1+n*(N-1)
    else
        error("Invalid configuration: $config.")
    end
    SlaterState(collect(pos), L)
end

# from an orbital matrix, optionally re-orthonormalized
"""
    SlaterState(B::AbstractMatrix; orthonormalize=false)

Wrap an orbital matrix directly. With `orthonormalize=true` the columns are
re-orthonormalized via QR.
"""
SlaterState(B::AbstractMatrix; orthonormalize::Bool=false) =
    SlaterState(orthonormalize ? Matrix(qr!(Matrix(B)).Q) : Matrix(B))

# ground state of a quadratic Hamiltonian
"""
    SlaterState(h::Hermitian, N)

Ground state at filling `N` of the quadratic Hamiltonian `H = ∑ hᵢⱼ c⁺ᵢ cⱼ`:
the lowest `N` single-particle eigen-orbitals.
"""
function SlaterState(h::Hermitian, N::Integer)
    vecs = eigvecs(h)            # ascending eigenvalues
    SlaterState(Matrix{ComplexF64}(vecs[:, 1:N]))
end

# random Slater determinant
"""
    SlaterState(L::Integer, N::Integer; rng=Random.default_rng())

Random Slater determinant of `N` particles on `L` modes (first `N` columns of a
Haar-random unitary).
"""
function SlaterState(L::Integer, N::Integer; rng::AbstractRNG=Random.default_rng())
    Q = Matrix(qr(randn(rng, ComplexF64, L, L)).Q)
    SlaterState(Matrix{ComplexF64}(Q[:, 1:N]))
end

#---------------------------------------------------------------------------------------------------
# Mixed state — correlation-matrix form
#---------------------------------------------------------------------------------------------------
"""
    CorrelationState{T} <: NumberConservingGaussianState{T}

General (possibly mixed) number-conserving Gaussian state, stored by its full
`L×L` correlation matrix `Cᵢⱼ = ⟨c⁺ᵢ cⱼ⟩` (Hermitian, spectrum ⊆ [0,1]). Use for
finite-temperature/Gibbs states and (later) Lindblad steady states. Pure states
are the special case `C² = C`.
"""
mutable struct CorrelationState{T<:Number} <: NumberConservingGaussianState{T}
    C::Hermitian{T,Matrix{T}}
end

"""
    CorrelationState(C::AbstractMatrix)

Wrap a correlation matrix (stored as `Hermitian`, element type `ComplexF64`).
"""
CorrelationState(C::AbstractMatrix) = CorrelationState(Hermitian(Matrix{ComplexF64}(C)))

"""
    CorrelationState(occ::AbstractVector{<:Real})

Diagonal correlation matrix from site occupations `occ ∈ [0,1]`.
"""
CorrelationState(occ::AbstractVector{<:Real}) =
    CorrelationState(Hermitian(Matrix{ComplexF64}(diagm(float.(occ)))))

"""
    CorrelationState(pos::AbstractVector{<:Integer}, L::Integer)

Fock state (as a correlation state) with sites `pos` occupied on `L` modes.
"""
function CorrelationState(pos::AbstractVector{<:Integer}, L::Integer)
    occ = zeros(L)
    occ[pos] .= 1
    CorrelationState(occ)
end

# pure -> mixed
CorrelationState(s::SlaterState) = CorrelationState(correlation_matrix(s))

"""
    thermalstate(h::Hermitian; β, μ=0)

Gibbs state of `H = ∑ hᵢⱼ c⁺ᵢ cⱼ` at inverse temperature `β` and chemical
potential `μ`: `C = U f(β(ε-μ)) U'` with Fermi–Dirac occupations.
"""
function thermalstate(h::Hermitian; β::Real, μ::Real=0)
    vals, U = eigen(h)
    occ = @. 1 / (1 + exp(β * (vals - μ)))
    CorrelationState(Hermitian(U * Diagonal(complex(occ)) * U'))
end

"""
    maximally_mixed(L; filling=0.5)

Maximally mixed state at the given uniform `filling`: `C = filling · I`.
"""
maximally_mixed(L::Integer; filling::Real=0.5) = CorrelationState(fill(float(filling), L))

#---------------------------------------------------------------------------------------------------
# Core accessors
#---------------------------------------------------------------------------------------------------
"""
    correlation_matrix(s) -> Matrix

Full correlation matrix `Cᵢⱼ = ⟨c⁺ᵢ cⱼ⟩` (Hermitian, eigenvalues ∈ [0,1]).
"""
correlation_matrix(s::SlaterState)      = conj(s.B) * transpose(s.B)
correlation_matrix(s::CorrelationState) = Matrix(s.C)

"""
    correlation_matrix(s, A) -> Matrix

Restricted block `C[A,A]` (the basis of entanglement diagnostics).
"""
correlation_matrix(s::NumberConservingGaussianState, A::AbstractVector{<:Integer}) =
    correlation_matrix(s)[A, A]
function correlation_matrix(s::SlaterState, A::AbstractVector{<:Integer})
    BA = view(s.B, A, :)
    conj(BA) * transpose(BA)
end

"""
    correlation(s, i, j) -> Number

Two-point function `⟨c⁺ᵢ cⱼ⟩`.
"""
correlation(s::SlaterState, i::Integer, j::Integer) = dot(view(s.B, i, :), view(s.B, j, :))
correlation(s::CorrelationState, i::Integer, j::Integer) = s.C[i, j]

"""
    nmodes(s) -> Int

Number of single-particle modes `L`.
"""
nmodes(s::SlaterState)      = size(s.B, 1)
nmodes(s::CorrelationState) = size(s.C, 1)

Base.eltype(::NumberConservingGaussianState{T}) where {T} = T
Base.copy(s::SlaterState)      = SlaterState(copy(s.B))
Base.copy(s::CorrelationState) = CorrelationState(Hermitian(Matrix(s.C)))

#---------------------------------------------------------------------------------------------------
# Purity test and pure -> orbital conversion
#---------------------------------------------------------------------------------------------------
"""
    ispure(s; tol=1e-8) -> Bool

Whether the state is pure (`C² ≈ C`, i.e. all occupations ≈ 0 or 1).
"""
ispure(::SlaterState) = true
function ispure(s::CorrelationState; tol::Real=1e-8)
    p = real.(eigvals(s.C))
    all(x -> abs(x) < tol || abs(1 - x) < tol, p)
end

"""
    SlaterState(s::CorrelationState; tol=1e-8)

Recover the orbital form of a pure correlation state (eigenvectors with
occupation ≈ 1 become the columns of `B`). Errors if `s` is not pure.
"""
function SlaterState(s::CorrelationState; tol::Real=1e-8)
    vals, U = eigen(s.C)
    occ = findall(x -> abs(real(x) - 1) < tol, vals)
    ispure(s; tol) || throw(ArgumentError("CorrelationState is not pure; cannot convert to SlaterState"))
    SlaterState(Matrix{ComplexF64}(U[:, occ]))
end
