# FlashBang.jl

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://DerangedIons.github.io/FlashBang.jl/stable/)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://DerangedIons.github.io/FlashBang.jl/dev/)
[![Build Status](https://github.com/DerangedIons/FlashBang.jl/actions/workflows/CI.yml/badge.svg?branch=main)](https://github.com/DerangedIons/FlashBang.jl/actions/workflows/CI.yml?query=branch%3Amain)
[![Coverage](https://codecov.io/gh/DerangedIons/FlashBang.jl/branch/main/graph/badge.svg)](https://codecov.io/gh/DerangedIons/FlashBang.jl)

Cardiac tissue electrophysiology on orthogonal, structured grids.

FlashBang solves the monodomain equation

```math
\begin{aligned}
\chi C_\mathrm{m} \partial_t \varphi_\mathrm{m}
    &= \nabla\cdot\kappa\nabla\varphi_\mathrm{m}
     + \chi\left(I_\mathrm{ion}(\varphi_\mathrm{m}, s, t) + I_\mathrm{stim}(x, t)\right) \\
\partial_t s &= f(\varphi_\mathrm{m}, s, t)
\end{aligned}
```

by reaction-diffusion operator splitting: [MatrixFreeOperators.jl](https://github.com/RallypointOne/MatrixFreeOperators.jl)
supplies the spatial operators, [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) the cell
kinetics, and [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl)
the splitting itself.

## What it does, and what it deliberately does not

Restricting the domain to **orthogonal structured grids** is the whole design. That
restriction is what makes the spatial operators matrix-free — no assembly, no sparse
factorization, no mesh data structure — and it is why the same source runs on a GPU without a
second implementation. Unstructured meshes, curved geometry, and finite elements are
[Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl)'s job, not this package's.

The pipeline is nevertheless *congruent* with Thunderbolt's by convention — same names, same
shapes, so a model description reads the same in either — while depending on none of it.

**Currently in:** monodomain, N-dimensional, isotropic or axis-aligned anisotropic
conductivity, zero-flux boundaries, arbitrary CytoZoo cell models, spatial parameter
heterogeneity, CPU and CUDA.

**Currently out:** unstructured meshes, bidomain, mechanics, spatially varying conductivity
fields, non-zero-flux boundaries, Rush–Larsen inner solvers, adaptive mesh refinement,
conduction-velocity and pseudo-ECG post-processing.

## The pipeline

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

# 4. Solve. Diffusion and reaction each get their own inner algorithm.
prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005)

while integrator.t < 400.0
    step!(integrator, 0.005, true)
    φ = getvariable(integrator.u, f, :φₘ)     # a view, not a copy
end
```

## Examples

`julia --project=examples -t auto examples/<name>.jl`

| | |
|---|---|
| `cable_1d.jl` | the snippet above with reporting and a figure — one FitzHugh–Nagumo cable, one propagating action potential |
| `spiral_2d.jl` | a reentrant spiral from an S1–S2 cross-field protocol, Aliev–Panfilov kinetics on an 80×80 sheet |
| `niederer_benchmark.jl` | the Niederer et al. (2011) N-version benchmark: a 3D anisotropic slab with ten Tusscher–Panfilov 2006 kinetics, activation times at nine reference points compared against an independent finite-difference solve |

The last two are ports of the corresponding MatrixFreeOperators.jl examples, which run the
same physics on an adaptive block forest through a hand-rolled Godunov split. Reading a pair
side by side is the fastest way to see what the pipeline buys and what the structured-grid
restriction costs.

Their cell models — `aliev_panfilov.jl` and `ten_tusscher_2006.jl` — live in `examples/`
rather than in CytoZoo because they are here to drive figures. Run
`julia --project=examples examples/ten_tusscher_2006.jl` on its own to check the TT06 port
against single-cell targets.

## The solution vector

State-blocked (structure of arrays), matching Thunderbolt's `StateBlockedLayout`:

```math
u = \left[\varphi_\mathrm{m}(1{:}N);\; s_1(1{:}N);\; s_2(1{:}N);\; \ldots;\; s_{M-1}(1{:}N)\right]
```

Every state is contiguous, so the diffusion half acts on the leading block `1:N` with no
gather and no synchronizer object, and neighbouring GPU threads read neighbouring nodes.
Reach into it by name rather than by index:

```julia
create_initial_condition(f)             # every node at the cell model's default state
getvariable(u, f, :φₘ)                  # view of the voltage block
getvariable(u, f, :states)              # view of every non-voltage state
setvariable!(u, f, :φₘ) do x            # impose a profile
    x[1] < 1.0 ? 1.0 : 0.0
end
```

## GPU

Put the backend on the grid and the whole pipeline follows — field allocation, the prepared
operator's scratch, node coordinates, and the initial condition:

```julia
using CUDA
grid = CartesianGrid(((0.0, 20.0),), (400,);
                     bc = ((Neumann(), Neumann()),), device = CUDABackend())
```

The reaction half then runs as one KernelAbstractions kernel over the nodes. The cell model
has to be isbits to ride into it — `CytoZoo.FHNModel` is; `ToRORd`, whose
parameters live in a heap `Vector`, is not yet.

## Precision

Use `Float64`. The types are generic and `Float32` runs, but the ArmyHeart reference measured
non-finite results across 58.6% of the ToRORd parameter space in single precision.

## Status

Pre-release, unregistered, and the API is expected to move. Install from source alongside its
unregistered dependencies:

```julia
using Pkg
Pkg.develop(url = "https://github.com/RallypointOne/MatrixFreeOperators.jl")
Pkg.develop(url = "https://github.com/DerangedIons/CytoZoo.jl")
Pkg.develop(url = "https://github.com/DerangedIons/FlashBang.jl")
```
