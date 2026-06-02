#---------------------------------------------------------------------------------------------------
# Quasi modes
#---------------------------------------------------------------------------------------------------
export QuasiMode
"""
    QuasiMode(I, V, L)

A single-particle ("quasi") mode

    d⁺ = ∑ⱼ Vⱼ c⁺_{Iⱼ}

on `L` sites, stored sparsely by the support indices `I` and amplitudes `V`.
For a normalized mode `‖V‖ = 1`, so that `d⁺d` is a projector.
"""
struct QuasiMode{T<:Number}
    I::Vector{Int64}
    V::Vector{T}
    L::Int64
    function QuasiMode(I::AbstractVector{<:Integer}, V::AbstractVector{T}, L::Integer) where T <: Number
        ind = [mod(i-1, L)+1 for i in I]
        new{T}(ind, V, Int64(L))
    end
end
#---------------------------------------------------------------------------------------------------
"""
    vector(qm::QuasiMode) -> SparseVector

Materialize the mode amplitudes as a length-`L` sparse vector.
"""
vector(qm::QuasiMode) = sparsevec(qm.I, qm.V, qm.L)
#---------------------------------------------------------------------------------------------------
"""
    inner!(out, qm, B) ; inner(qm, B)

Overlaps of the mode `d` with each occupied orbital (column of `B`):
`outₖ = ⟨d | Bₖ⟩ = ∑ⱼ conj(Vⱼ) B[Iⱼ, k]`.
"""
function inner!(out::AbstractVector, qm::QuasiMode, B::AbstractMatrix)
    turbo_dot!(out, qm.V, B, qm.I)
end
function inner(qm::QuasiMode, B::AbstractMatrix)
    out = Vector{ComplexF64}(undef, size(B, 2))
    inner!(out, qm, B)
    out
end
