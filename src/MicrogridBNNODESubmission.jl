module MicrogridBNNODESubmission

# Minimal module to support Pkg precompile/test.
# Core functionality is organized in scripts/ and src/ submodules
# (e.g., Microgrid, NeuralNODEArchitectures, StatisticalFramework).

include("microgrid_system.jl")
include("neural_ode_architectures.jl")
include("training.jl")
include("symbolic_extraction.jl")

end 