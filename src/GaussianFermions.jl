module GaussianFermions

using LinearAlgebra, SparseArrays, KrylovKit, DifferentialEquations, StaticArrays, LoopVectorization
import Base.:*

include("FreeFermion.jl")
include("Quadratic.jl")


end # module GaussianFermions
