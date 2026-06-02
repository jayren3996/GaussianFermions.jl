#---------------------------------------------------------------------------------------------------
# Number-conserving quadratic Hamiltonians
#---------------------------------------------------------------------------------------------------
export AbstractQuadraticHamiltonian, QuadraticHamiltonian
export hopping, chemical_potential, propagator

abstract type AbstractQuadraticHamiltonian end

"""
    QuadraticHamiltonian{T} <: AbstractQuadraticHamiltonian

A number-conserving quadratic Hamiltonian `H = ∑ᵢⱼ hᵢⱼ c⁺ᵢ cⱼ`, stored by its
`L×L` Hermitian single-particle matrix `h`. Caches the last propagator
`exp(-i h dt)` so repeated fixed-`dt` stepping does not re-diagonalize.
"""
mutable struct QuadraticHamiltonian{T<:Complex} <: AbstractQuadraticHamiltonian
    h::Hermitian{T,Matrix{T}}
    cache::Union{Nothing,Tuple{Float64,Matrix{T}}}
end

"""
    QuadraticHamiltonian(h::AbstractMatrix)

Wrap a single-particle matrix `h` (Hermitized, element type `ComplexF64`).
"""
QuadraticHamiltonian(h::AbstractMatrix) =
    QuadraticHamiltonian{ComplexF64}(Hermitian(Matrix{ComplexF64}(h)), nothing)

"""
    hopping(L; J=1.0, pbc=false)

Nearest-neighbour hopping chain `H = ∑ᵢ J c⁺ᵢ cᵢ₊₁ + h.c.` on `L` sites.
"""
function hopping(L::Integer; J::Number=1.0, pbc::Bool=false)
    h = zeros(ComplexF64, L, L)
    for i in 1:L-1
        h[i, i+1] = J
        h[i+1, i] = conj(J)
    end
    if pbc
        h[L, 1] += J
        h[1, L] += conj(J)
    end
    QuadraticHamiltonian(h)
end

"""
    hopping(edges, L)

Generic hopping `H = ∑ tᵢⱼ c⁺ᵢ cⱼ + h.c.` from a list of `(i, j, t)` tuples.
"""
function hopping(edges::AbstractVector{<:Tuple}, L::Integer)
    h = zeros(ComplexF64, L, L)
    for (i, j, t) in edges
        h[i, j] += t
        h[j, i] += conj(t)
    end
    QuadraticHamiltonian(h)
end

"""
    chemical_potential(μ::AbstractVector)

Diagonal on-site term `H = ∑ᵢ μᵢ nᵢ`.
"""
chemical_potential(μ::AbstractVector) = QuadraticHamiltonian(diagm(ComplexF64.(μ)))

Base.:+(a::QuadraticHamiltonian, b::QuadraticHamiltonian) =
    QuadraticHamiltonian(Matrix(a.h) + Matrix(b.h))

nmodes(H::QuadraticHamiltonian) = size(H.h, 1)
Base.Matrix(H::QuadraticHamiltonian) = Matrix(H.h)

"""
    propagator(H::QuadraticHamiltonian, dt) -> Matrix

The single-particle propagator `U = exp(-i h dt)` (memoized on `H` for the last `dt`).
"""
function propagator(H::QuadraticHamiltonian, dt::Real)
    if H.cache !== nothing && H.cache[1] == Float64(dt)
        return H.cache[2]
    end
    U = evo_operator(H.h, dt)
    H.cache = (Float64(dt), U)
    U
end
