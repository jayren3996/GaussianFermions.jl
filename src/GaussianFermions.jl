module GaussianFermions

using LinearAlgebra, SparseArrays, KrylovKit, DifferentialEquations, StaticArrays, LoopVectorization, Random
import Base.:*

# --- number-conserving (U(1)) layer ---
include("LinAlg.jl")        # numerical kernels (unchanged)
include("Modes.jl")         # QuasiMode + overlaps
include("States.jl")        # SlaterState / CorrelationState
include("Observables.jl")   # densities, entropies, diagnostics
include("Hamiltonians.jl")  # QuadraticHamiltonian + propagator
include("Channels.jl")      # dissipation / measurement channels
include("Trajectory.jl")    # evolution, trajectory engine, ensemble runner
include("CorrelationLindblad.jl")
include("MajoranaStates.jl")
include("BdGHamiltonians.jl")

# --- Majorana / BdG covariance layer (stage 2; unchanged) ---
include("Quadratic.jl")

end # module GaussianFermions
