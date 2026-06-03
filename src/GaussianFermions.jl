module GaussianFermions

using LinearAlgebra, SparseArrays, StaticArrays, LoopVectorization, Random

# --- number-conserving (U(1)) layer ---
include("LinAlg.jl")        # numerical kernels (unchanged)
include("Modes.jl")         # QuasiMode + overlaps
include("States.jl")        # SlaterState / CorrelationState
include("Observables.jl")   # densities, entropies, diagnostics
include("Hamiltonians.jl")  # QuadraticHamiltonian + propagator
include("Channels.jl")      # dissipation / measurement channels
include("Trajectory.jl")    # evolution, gates, quantum-trajectory primitives
include("CorrelationLindblad.jl")
# --- Majorana / BdG covariance layer ---
include("MajoranaStates.jl")
include("BdGHamiltonians.jl")
include("MajoranaObservables.jl")
include("MajoranaLindblad.jl")
include("MajoranaTrajectory.jl")

end # module GaussianFermions
