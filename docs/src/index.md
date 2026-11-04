```@raw html
---
layout: home

hero:
  name: FlashBang.jl
  text: Cardiac Tissue Electrophysiology
  tagline: The monodomain equation on orthogonal structured grids — matrix-free, SciML-native, and the same source on CPU or GPU.
  actions:
    - theme: brand
      text: Get Started
      link: /getting_started
    - theme: alt
      text: Run on a GPU
      link: /guides/gpu
    - theme: alt
      text: View on GitHub
      link: https://github.com/DerangedIons/FlashBang.jl

features:
  - icon: ▦
    title: Structured Grids by Design
    details: Restricting the domain to orthogonal grids is what makes the spatial operators matrix-free — no assembly, no sparse factorization, no mesh data structure.
  - icon: ⇄
    title: An Ordinary SciML Solve
    details: Reaction-diffusion operator splitting through OrdinaryDiffEqOperatorSplitting. Problems, solutions, callbacks, and symbolic indexing work as they do anywhere in SciML.
  - icon: ⚡
    title: One Source, CPU or GPU
    details: Put a backend on the grid and the whole pipeline follows. The reaction half becomes one KernelAbstractions kernel over the nodes.
---
```

## What it solves

FlashBang solves the monodomain equation

```math
\begin{aligned}
\chi C_\mathrm{m} \partial_t \varphi_\mathrm{m}
    &= \nabla\cdot\kappa\nabla\varphi_\mathrm{m}
     + \chi\left(I_\mathrm{ion}(\varphi_\mathrm{m}, s, t) + I_\mathrm{stim}(x, t)\right) \\
\partial_t s &= f(\varphi_\mathrm{m}, s, t)
\end{aligned}
```

by reaction-diffusion operator splitting. [MatrixFreeOperators.jl](https://github.com/RallypointOne/MatrixFreeOperators.jl) supplies the spatial operators, [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) the cell kinetics, and [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl) the splitting.

Restricting the domain to orthogonal structured grids is the design, not a limitation waiting to be lifted: it is what makes the spatial operators matrix-free and what lets the same source run on a GPU. Unstructured meshes are [Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl)'s job. Building a problem nevertheless uses Thunderbolt's names and shapes, while depending on none of it; solving it is an ordinary SciML `solve`. Driving the integrator by hand, the way Thunderbolt's tutorials do, is covered in [Thunderbolt compatibility](guides/thunderbolt.md).

## Quick start

```julia
using FlashBang
using CytoZoo: FHNModel
using OrdinaryDiffEqLowOrderRK: Euler

grid = CartesianGrid(((0.0, 20.0),), (400,); bc = ((Neumann(), Neumann()),))

stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)

model = MonodomainModel(; κ = 0.1, ion = FHNModel(), stim)
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)

prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, saveat = 1.0)

sol[:φₘ, end]              # voltage along the cable at t = 400
```

[Getting Started](getting_started.md) walks through each line.

## Scope

**Currently in:** monodomain, N-dimensional, isotropic or axis-aligned anisotropic conductivity, zero-flux boundaries, arbitrary CytoZoo cell models, spatial parameter heterogeneity, CPU and CUDA.

**Currently out:** unstructured meshes, bidomain, mechanics, spatially varying conductivity fields, non-zero-flux boundaries, Rush–Larsen inner solvers, adaptive mesh refinement, conduction-velocity and pseudo-ECG post-processing.
