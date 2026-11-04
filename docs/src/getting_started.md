```@meta
CurrentModule = FlashBang
```

# Getting Started

## Installation

FlashBang is pre-release and unregistered, and the API is expected to move. Install it from source alongside its unregistered dependencies:

```julia
using Pkg
Pkg.develop(url = "https://github.com/RallypointOne/MatrixFreeOperators.jl")
Pkg.develop(url = "https://github.com/DerangedIons/CytoZoo.jl")
Pkg.develop(url = "https://github.com/DerangedIons/FlashBang.jl")
```

The examples below also use `OrdinaryDiffEqLowOrderRK` for the inner `Euler` steppers.

## The pipeline

Four objects, in order.

| Step | Type | What it says |
|---|---|---|
| 1 | [`MonodomainModel`](@ref) | the continuous problem: `κ`, `χ`, `Cₘ`, the cell model, the stimulus |
| 2 | [`ReactionDiffusionSplit`](@ref) | how to split it, and any spatial parameter heterogeneity |
| 3 | [`FiniteDifferenceDiscretization`](@ref) + a `CartesianGrid` | where to solve it |
| 4 | [`semidiscretize`](@ref) | assembles a `GenericSplitFunction` ready for `OperatorSplittingProblem` |

Then an `OperatorSplittingProblem` and `solve`, as with any SciML problem.

## A first solve

One FitzHugh–Nagumo cable, stimulated at its left end, carrying one propagating action potential:

```julia
using FlashBang
using CytoZoo: FHNModel
using OrdinaryDiffEqLowOrderRK: Euler

# 1. A grid. Cell-centered, uniform, zero flux at both ends.
grid = CartesianGrid(((0.0, 20.0),), (400,); bc = ((Neumann(), Neumann()),))

# 2. A stimulus. Positive is depolarizing — the PDE-form convention, which is the
#    opposite of the cell-model-internal one. `nonzero_intervals` lets the right-hand
#    side skip the evaluation once the pulse is over.
stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)

# 3. The continuous problem, then how to split it, then the semidiscretization.
model = MonodomainModel(; κ = 0.1, ion = FHNModel(), stim)
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)

# 4. Solve. Diffusion and reaction each get their own inner algorithm. The split is
#    fixed-step, so `dt` is required; `saveat` says which states to keep.
prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, saveat = 1.0)

# 5. Read states by name, as with any SciML solution.
sol[:φₘ, end]              # voltage along the cable at t = 400
sol[:φₘ]                   # ... at every saved time: a vector of length-400 vectors
sol(250.5; idxs = :φₘ)     # between saves, linearly interpolated
```

The reaction half is threaded over nodes on the CPU, so start Julia with `-t auto`.

## Where to next

- [The Solution Vector](guides/solution_vector.md) — how the state is laid out and how to reach into it by name.
- [Saving Output](guides/saving.md) — why saving is opt-in, and how to record per-step quantities without saving.
- [Running on a GPU](guides/gpu.md) — the one-keyword switch, and what has to be true of your model for it to work.
- [Things That Will Bite You](reference/pitfalls.md) — read this before your first real run.
