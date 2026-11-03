"""
    FlashBang

Cardiac tissue electrophysiology on orthogonal, structured grids.

Building a problem uses Thunderbolt's names and shapes (with no dependency on it); solving
it is an ordinary SciML `solve`:

```julia
model = MonodomainModel(; κ, ion)                       # continuous problem
split = ReactionDiffusionSplit(model)                   # how to split it
f     = semidiscretize(split, FiniteDifferenceDiscretization(), grid)
prob  = OperatorSplittingProblem(f, create_initial_condition(f), tspan)
sol   = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt, saveat)
sol[:φₘ, end]                                           # states by name
```

Spatial operators come from MatrixFreeOperators.jl and cell kinetics from CytoZoo.jl.
Restricting the domain to orthogonal structured grids is what buys the matrix-free
operators; unstructured meshes are Thunderbolt's job, not this package's.

See also: [`MonodomainModel`](@ref), [`semidiscretize`](@ref),
[`create_initial_condition`](@ref).
"""
module FlashBang

using Adapt: Adapt
using LinearAlgebra: mul!
using StaticArrays: SVector

using KernelAbstractions: KernelAbstractions
using KernelAbstractions: @index, @kernel

using CytoZoo: CytoZoo

# Explicit imports throughout: MatrixFreeOperators exports `solve` and `gradient`, which
# would clash with the SciML stack and with Base. The `solve` FlashBang exports is
# SciMLBase's, so a user who writes `using FlashBang, MatrixFreeOperators` must qualify it.
using MatrixFreeOperators:
    CartesianGrid,
    Dirichlet,
    Neumann,
    Periodic,
    boundary_conditions,
    cell_center,
    derivative,
    dimension,
    interior,
    laplacian,
    local_size,
    prepare,
    scalar_field,
    spacing

using OrdinaryDiffEqOperatorSplitting:
    GenericSplitFunction, LieTrotterGodunov, OperatorSplittingProblem, StrangMarchuk

using SciMLBase: SciMLBase, DiscreteCallback, ODEFunction, init, solve, solve!, step!
using SymbolicIndexingInterface: SymbolicIndexingInterface as SII

include("stimulus.jl")
include("models.jl")
include("discretization.jl")
include("reaction.jl")
include("semidiscretize.jl")
include("solution.jl")
include("observer.jl")

# FlashBang's own vocabulary
export AbstractEPModel, MonodomainModel, ReactionDiffusionSplit
export AbstractStimulationProtocol, NoStimulationProtocol, TransmembraneStimulationProtocol
export FiniteDifferenceDiscretization, semidiscretize, diffusion_operator, node_coordinates
export create_initial_condition,
    getvariable, setvariable!, solution_size, num_nodes, variable_range

# Re-exported mesh vocabulary, so `using FlashBang` is enough to build a problem.
export CartesianGrid, Dirichlet, Neumann, Periodic
export boundary_conditions, cell_center, dimension, interior, local_size, spacing

# Re-exported splitting vocabulary, for the same reason. `solve` is the main entry point:
# it returns an `ODESolution` that indexes by state name (`sol[:φₘ]`) through the
# symbolic-indexing hook in `solution.jl`. `init`/`step!`/`solve!` drive the integrator by
# hand, the Thunderbolt-style path.
export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov, StrangMarchuk
export init, solve, solve!, step!
export StateObserver

end # module
