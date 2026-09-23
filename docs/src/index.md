```@meta
CurrentModule = FlashBang
```

# FlashBang.jl

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

by reaction-diffusion operator splitting. [MatrixFreeOperators.jl](https://github.com/RallypointOne/MatrixFreeOperators.jl)
supplies the spatial operators, [CytoZoo.jl](https://github.com/DerangedIons/CytoZoo.jl) the
cell kinetics, and [OrdinaryDiffEqOperatorSplitting.jl](https://github.com/SciML/OrdinaryDiffEqOperatorSplitting.jl)
the splitting.

Restricting the domain to orthogonal structured grids is the design, not a limitation waiting
to be lifted: it is what makes the spatial operators matrix-free and what lets the same source
run on a GPU. Unstructured meshes are [Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl)'s
job. FlashBang's pipeline is nevertheless congruent with Thunderbolt's by convention — same
names, same shapes — while depending on none of it.

## The pipeline

Four objects, in order.

| Step | Type | What it says |
|---|---|---|
| 1 | [`MonodomainModel`](@ref) | the continuous problem: `κ`, `χ`, `Cₘ`, the cell model, the stimulus |
| 2 | [`ReactionDiffusionSplit`](@ref) | how to split it, and any spatial parameter heterogeneity |
| 3 | [`FiniteDifferenceDiscretization`](@ref) + a `CartesianGrid` | where to solve it |
| 4 | [`semidiscretize`](@ref) | assembles a `GenericSplitFunction` ready for `OperatorSplittingProblem` |

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
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005)

while integrator.t < 400.0
    step!(integrator, 0.005, true)
    φ = getvariable(integrator.u, f, :φₘ)
end
```

## Examples

Run with `julia --project=examples -t auto examples/<name>.jl`.

| Example | What it shows |
|---|---|
| `cable_1d.jl` | the snippet above with reporting and a figure — one FitzHugh–Nagumo cable, one propagating action potential |
| `spiral_2d.jl` | a reentrant spiral from an S1–S2 cross-field protocol, Aliev–Panfilov kinetics on an 80×80 sheet |
| `niederer_benchmark.jl` | the Niederer et al. (2011) N-version benchmark — a 3D anisotropic slab with ten Tusscher–Panfilov 2006 kinetics, activation times at nine reference points against an independent finite-difference solve |

The last two are ports of the corresponding MatrixFreeOperators.jl examples, which run the
same physics on an adaptive block forest through a hand-rolled Godunov split. Reading a pair
side by side shows what the pipeline buys and what the structured-grid restriction costs —
FlashBang carries every cell at the finest spacing, where the forest carries roughly a fifth
of them.

Their cell models live in `examples/` rather than in CytoZoo because they exist to drive
figures. `examples/ten_tusscher_2006.jl` doubles as its own acceptance test: run it directly
to check the 19-state port against published single-cell targets.

## The solution vector

State-blocked (structure of arrays), matching Thunderbolt's `StateBlockedLayout`:

```math
u = \left[\varphi_\mathrm{m}(1{:}N);\; s_1(1{:}N);\; s_2(1{:}N);\; \ldots;\; s_{M-1}(1{:}N)\right]
```

Every state is contiguous, so the diffusion half acts on the leading block `1:N` — no gather,
no synchronizer object, and coalesced access on a GPU. Reach into it by name with
[`getvariable`](@ref), [`setvariable!`](@ref), and [`variable_range`](@ref) rather than by
index.

## Things that will bite you

- **The stimulus sign is inverted relative to a cell model.** A protocol returns `Iₐₚₚ` in
  PDE form, where positive depolarizes; a CytoZoo cell model's internal stimulus is a
  membrane current, where negative depolarizes. See [`AbstractStimulationProtocol`](@ref).
  Stimulation belongs to the diffusion half — the cell model's own stimulus must be off, and
  [`MonodomainModel`](@ref) warns when it can tell that it is not.
- **The grid is cell-centered.** `n` cells across a length `L` put the first node at `Δx/2`,
  not at `0`. Comparisons against a vertex-centered reference differ at `O(Δx)`.
- **Only zero-flux boundaries.** Anything else raises an `ArgumentError` naming the face.
- **One semidiscretization per concurrent solve.** The prepared operator inside the diffusion
  half owns scratch buffers and is stateful and single-threaded.
- **The splitting integrator has no SciML solution interface.** There is no `sol` to hand
  back; step it, or sample it with `SciMLIterators.TimeChoiceIterator`.
- **Use `Float64`.** `Float32` runs, but the ArmyHeart reference measured non-finite results
  across 58.6% of the ToRORd parameter space in single precision.

## GPU

Put the backend on the grid — `CartesianGrid(…; device = CUDABackend())` — and field
allocation, the prepared operator's scratch, [`node_coordinates`](@ref), and
[`create_initial_condition`](@ref) all follow. The reaction half then runs as one
KernelAbstractions kernel over the nodes, which requires the cell model to be isbits.

## API

```@index
```

```@autodocs
Modules = [FlashBang]
```
