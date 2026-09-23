"""
    FlashBang

Cardiac tissue electrophysiology on orthogonal, structured grids.

The pipeline mirrors Thunderbolt's by convention — same names, same shapes, no dependency:

```julia
model = MonodomainModel(; κ, ion)                       # continuous problem
split = ReactionDiffusionSplit(model)                   # how to split it
f     = semidiscretize(split, FiniteDifferenceDiscretization(), grid)
prob  = OperatorSplittingProblem(f, create_initial_condition(f), tspan)
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt)
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
# would clash with the SciML stack and with Base.
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

using SciMLBase: ODEFunction, init, solve!, step!

include("stimulus.jl")
include("models.jl")
include("discretization.jl")
include("reaction.jl")
include("semidiscretize.jl")
include("solution.jl")

# FlashBang's own vocabulary
export AbstractEPModel, MonodomainModel, ReactionDiffusionSplit
export AbstractStimulationProtocol, NoStimulationProtocol, TransmembraneStimulationProtocol
export FiniteDifferenceDiscretization, semidiscretize, diffusion_operator, node_coordinates
export create_initial_condition,
    getvariable, setvariable!, solution_size, num_nodes, variable_range

# Re-exported mesh vocabulary, so `using FlashBang` is enough to build a problem.
export CartesianGrid, Dirichlet, Neumann, Periodic
export boundary_conditions, cell_center, dimension, interior, local_size, spacing

# Re-exported splitting vocabulary, for the same reason. `init`/`step!`/`solve!` but
# deliberately not `solve`: the operator-splitting integrator does not implement the SciML
# solution interface, so there is no `sol` object to hand back. Sample it over time with
# `SciMLIterators.TimeChoiceIterator(integrator, ts)` instead.
export GenericSplitFunction, OperatorSplittingProblem, LieTrotterGodunov, StrangMarchuk
export init, solve!, step!

end # module
