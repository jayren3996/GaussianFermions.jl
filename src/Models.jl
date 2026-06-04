#---------------------------------------------------------------------------------------------------
# Lightweight finite model constructors and spectral helpers
#---------------------------------------------------------------------------------------------------
export ssh_chain, aubry_andre_chain, kitaev_chain

function _check_model_length(L::Integer, name::String)
    L ≥ 1 || throw(ArgumentError("$name requires L ≥ 1, got $L"))
    Int(L)
end

"""
    ssh_chain(L; t1=1.0, t2=0.5, pbc=false) -> QuadraticHamiltonian

Finite Su-Schrieffer-Heeger chain with alternating nearest-neighbour hoppings.
Bond `(j, j+1)` uses `t1` for odd `j` and `t2` for even `j`. Periodic boundaries
require even `L` so the dimerization pattern closes consistently.
"""
function ssh_chain(L::Integer; t1::Number=1.0, t2::Number=0.5, pbc::Bool=false)
    L = _check_model_length(L, "ssh_chain")
    pbc && isodd(L) &&
        throw(ArgumentError("ssh_chain with pbc=true requires even L, got $L"))

    edges = Tuple{Int,Int,Number}[]
    for j in 1:L-1
        push!(edges, (j, j + 1, isodd(j) ? t1 : t2))
    end
    pbc && push!(edges, (L, 1, t2))
    hopping(edges, L)
end

"""
    aubry_andre_chain(L; J=1.0, λ=1.0, β=(sqrt(5)-1)/2, ϕ=0.0, pbc=false)

Finite Aubry-Andre chain with nearest-neighbour hopping `J` and onsite potential
`λ cos(2π β j + ϕ)`.
"""
function aubry_andre_chain(L::Integer; J::Number=1.0, λ::Real=1.0,
                           β::Real=(sqrt(5) - 1) / 2, ϕ::Real=0.0,
                           pbc::Bool=false)
    L = _check_model_length(L, "aubry_andre_chain")
    pbc && L == 1 &&
        throw(ArgumentError("aubry_andre_chain with pbc=true requires L ≥ 2, got $L"))

    onsite = [λ * cos(2π * β * j + ϕ) for j in 1:L]
    hopping(L; J, pbc) + chemical_potential(onsite)
end

"""
    kitaev_chain(L; t=1.0, Δ=1.0, μ=0.0, pbc=false) -> BdGHamiltonian

Finite spinless p-wave superconducting chain in the package BdG block convention:
`A[j,j] = -μ`, nearest-neighbour hopping `-t`, and antisymmetric pairing block
`B[j,j+1] = Δ`.
"""
function kitaev_chain(L::Integer; t::Number=1.0, Δ::Number=1.0, μ::Real=0.0,
                      pbc::Bool=false)
    L = _check_model_length(L, "kitaev_chain")
    pbc && L < 3 &&
        throw(ArgumentError("kitaev_chain with pbc=true requires L ≥ 3, got $L"))

    A = zeros(ComplexF64, L, L)
    B = zeros(ComplexF64, L, L)
    for j in 1:L
        A[j, j] = -μ
    end
    for j in 1:L-1
        A[j, j+1] = -t
        A[j+1, j] = -conj(t)
        B[j, j+1] = Δ
        B[j+1, j] = -Δ
    end
    if pbc && L > 1
        A[L, 1] = -t
        A[1, L] = -conj(t)
        B[L, 1] = Δ
        B[1, L] = -Δ
    end
    BdGHamiltonian(A, B)
end

export energy_spectrum, bloch_bands

"""
    energy_spectrum(H::QuadraticHamiltonian) -> Vector{Float64}
    energy_spectrum(H::BdGHamiltonian) -> Vector{Float64}

Dense finite-system spectrum. For `QuadraticHamiltonian`, returns the single-particle
energies. For `BdGHamiltonian`, returns the non-negative quasiparticle energies.
"""
energy_spectrum(H::QuadraticHamiltonian) = sort(real.(eigvals(Hermitian(Matrix(H)))))
energy_spectrum(H::BdGHamiltonian) = quasiparticle_energies(H)

function _bloch_energy_values(H::AbstractMatrix)
    Hmat = Matrix{ComplexF64}(H)
    size(Hmat, 1) == size(Hmat, 2) ||
        throw(ArgumentError("matrix-valued Bloch Hamiltonians must be square, got $(size(Hmat))"))
    isapprox(Hmat, Hmat'; atol=1e-10) ||
        throw(ArgumentError("matrix-valued Bloch Hamiltonians must be Hermitian"))
    vals = eigvals(Hermitian(Hmat))
    sort(real.(vals))
end
_bloch_energy_values(H::QuadraticHamiltonian) = energy_spectrum(H)
_bloch_energy_values(H::BdGHamiltonian) = energy_spectrum(H)
_bloch_energy_values(H) =
    throw(ArgumentError("Hk(k) must return a Hermitian matrix, QuadraticHamiltonian, or BdGHamiltonian; got $(typeof(H))"))

"""
    bloch_bands(Hk, kgrid) -> Matrix{Float64}

Evaluate a Bloch Hamiltonian function `Hk(k)` on `kgrid` and return a matrix whose
rows correspond to momenta and columns correspond to sorted bands. `Hk(k)` may
return a Hermitian matrix, `QuadraticHamiltonian`, or `BdGHamiltonian`. For BdG
Hamiltonians, the returned bands are the non-negative quasiparticle half-spectrum,
not the full particle-hole-symmetric Nambu spectrum.
"""
function bloch_bands(Hk, kgrid)
    ks = collect(kgrid)
    isempty(ks) && return zeros(Float64, 0, 0)
    values = [_bloch_energy_values(Hk(k)) for k in ks]
    nbands = length(first(values))
    all(v -> length(v) == nbands, values) ||
        throw(ArgumentError("bloch_bands requires Hk(k) to return the same number of bands for every k"))

    bands = zeros(Float64, length(ks), nbands)
    for (row, vals) in enumerate(values)
        bands[row, :] .= vals
    end
    bands
end
