#---------------------------------------------------------------------------------------------------
# Majorana / BdG observables
#
# Densities, particle number, and entanglement diagnostics for `MajoranaState`, built
# from the real antisymmetric covariance matrix Γ. The covariance entanglement
# spectrum {ν} ⊂ [0,1] plays the role the occupation spectrum plays for the
# number-conserving states, via the occupation probabilities (1 ± ν)/2.
#---------------------------------------------------------------------------------------------------

density(s::MajoranaState) = real.(diag(normal_correlation(s)))
# ⟨nᵢ⟩ = (1 - Γ[i, i+L])/2 directly from the covariance (O(1); avoids rebuilding C).
function density(s::MajoranaState, i::Integer)
    L = nmodes(s)
    1 ≤ i ≤ L || throw(ArgumentError("mode index $i out of range 1:$L"))
    (1 - s.Gamma[i, i+L]) / 2
end
particle_number(s::MajoranaState) = sum(density(s))
particle_number(s::MajoranaState, A) = sum(density(s, i) for i in A)

function _majorana_region_covariance(s::MajoranaState, A::AbstractVector{<:Integer})
    L = nmodes(s)
    inds = vcat(collect(A), collect(A) .+ L)
    covariance_matrix(s)[inds, inds]
end

function entanglement_spectrum(s::MajoranaState, A::AbstractVector{<:Integer})
    G = _majorana_region_covariance(s, A)
    n = length(A)
    vals = sort(real.(eigvals(Hermitian(1im * G))))
    clamp.(vals[n+1:end], 0.0, 1.0)
end

function entanglement_entropy(s::MajoranaState, A::AbstractVector{<:Integer})
    sum(entanglement_spectrum(s, A)) do ν
        _binary_shannon((1 + ν) / 2)
    end
end

function renyi_entropy(s::MajoranaState, A::AbstractVector{<:Integer}; α::Real=2)
    α == 1 && return entanglement_entropy(s, A)
    ν = entanglement_spectrum(s, A)
    p = @. (1 + ν) / 2
    isinf(α) && return -sum(log(max(pk, 1 - pk)) for pk in p)
    (1 / (1 - α)) * sum(log(pk^α + (1 - pk)^α) for pk in p)
end

function mutual_information(s::MajoranaState,
                            A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer}; α::Real=1)
    S(R) = renyi_entropy(s, R; α)
    S(A) + S(B) - S(vcat(A, B))
end

function tripartite_information(s::MajoranaState,
                                A::AbstractVector{<:Integer}, B::AbstractVector{<:Integer},
                                C::AbstractVector{<:Integer}; α::Real=1)
    S(R) = renyi_entropy(s, R; α)
    S(A) + S(B) + S(C) - S(vcat(A, B)) - S(vcat(A, C)) - S(vcat(B, C)) + S(vcat(A, B, C))
end
