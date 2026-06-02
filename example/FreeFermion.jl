include("../src/GaussianFermions.jl")

using LinearAlgebra, SparseArrays, Main.GaussianFermions

function dephasing(;
    L::Integer, γ::Real, d::Vector{<:Number}
)
    H = diagm(1=>ones(L-1),-1=>ones(L-1)) |> sparse |> majoranaform
    L = sparse([1,])
    M = begin
        Ms = map(1:N) 
        [spzeros(ComplexF64,N,N) for i in 1:N]
        for i in 1:N-1
            d = spzeros(ComplexF64,N)
            d[i] = 1/sqrt(2)
            d[i+1] = -1im/sqrt(2)
            Ms[i] = d*d'
            Ms
        end
        majoranaform.(Ms)./4
    end
    L = spzeros(2N,2N)
    quadraticlindblad(H,L,sqrt(γ).*M)
end