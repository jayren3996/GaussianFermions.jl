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
