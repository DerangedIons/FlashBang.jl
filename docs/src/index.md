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
job. Building a problem nevertheless uses Thunderbolt's names and shapes, while depending on
none of it; solving it is an ordinary SciML `solve`. Driving the integrator by hand, the way
Thunderbolt's tutorials do, is covered in [Thunderbolt compatibility](@ref).

## The pipeline

Four objects, in order.

| Step | Type | What it says |
|---|---|---|
| 1 | [`MonodomainModel`](@ref) | the continuous problem: `κ`, `χ`, `Cₘ`, the cell model, the stimulus |
| 2 | [`ReactionDiffusionSplit`](@ref) | how to split it, and any spatial parameter heterogeneity |
| 3 | [`FiniteDifferenceDiscretization`](@ref) + a `CartesianGrid` | where to solve it |
| 4 | [`semidiscretize`](@ref) | assembles a `GenericSplitFunction` ready for `OperatorSplittingProblem` |

Then an `OperatorSplittingProblem` and `solve`, as with any SciML problem.

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

# The split is fixed-step, so `dt` is required; `saveat` says which states to keep.
prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, saveat = 1.0)

sol[:φₘ, end]              # voltage along the cable at t = 400
sol[:φₘ]                   # ... at every saved time: a vector of length-400 vectors
sol(250.5; idxs = :φₘ)     # between saves, linearly interpolated
```

## Examples

Run with `julia --project=examples -t auto examples/<name>.jl`.

| Example | What it shows |
|---|---|
| `cable_1d.jl` | the snippet above with reporting and a figure — one FitzHugh–Nagumo cable, one propagating action potential, activation times recorded by a [`StateObserver`](@ref) |
| `spiral_2d.jl` | a reentrant spiral from an S1–S2 cross-field protocol, Aliev–Panfilov kinetics on an 80×80 sheet |
| `niederer_benchmark.jl` | the Niederer et al. (2011) N-version benchmark — a 3D anisotropic slab with ten Tusscher–Panfilov 2006 kinetics, activation times at nine reference points against an independent finite-difference solve |
| `niederer_animation.jl` | the same slab as a video, with accumulating isochrones — nothing saved, every frame observed in flight |

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
no synchronizer object, and coalesced access on a GPU. Reach into it by name rather than by
index.

On a solution, symbolic indexing works as it does for any SciML solution, with each name
standing for a whole block of `N` values:

```julia
sol[:φₘ, end]                # copy of the voltage block at the last saved time
sol[:states, end]            # every non-voltage state, length N*(M-1)
sol[:CaMKt]                  # one named cell-model state, at every saved time
getvariable(sol, f, :φₘ)     # views instead of copies, one per saved time
```

The names are the cell model's state names, the model's `φ_symbol` (`:φₘ`, aliasing whatever
the cell model calls its voltage), and its `state_symbol` (`:states`). The same names work on
the problem (`prob[:φₘ]`) and on an integrator (`integrator[:φₘ]`).

On a raw state vector — an initial condition, an `integrator.u`, a `sol.u[k]` — use
[`getvariable`](@ref), [`setvariable!`](@ref), and [`variable_range`](@ref):

```julia
u₀ = create_initial_condition(f)       # every node at the cell model's default state
setvariable!(u₀, f, :φₘ) do x          # impose a profile
    x[1] < 1.0 ? 1.0 : 0.0
end
getvariable(u₀, f, :φₘ)                # view of the voltage block
```

## Saving output

Saving works differently from OrdinaryDiffEq's defaults, in ways that matter for a field
problem:

- **It is opt-in.** Without `saveat`, only the first and last states are kept.
- **Every save is the whole state** — every cell-model state at every node. `save_idxs` is
  not supported: it does not shrink saves.
- **A `saveat` time that falls inside a step is linearly interpolated.** Pass the same times
  as `tstops` to make a step land on each one. Accumulated rounding in `t` can then force an
  extra, near-zero step to reach a stop, which shifts a step-counted [`StateObserver`](@ref)
  off its time grid. When the save times are multiples of `dt`, leave `tstops` out.

What one save costs, in `Float64`:

| Problem | Nodes × states | One save |
|---|---|---|
| `cable_1d.jl` | 400 × 2 | 6.4 KB |
| `spiral_2d.jl` | 256² × 2 | 1.05 MB |
| `niederer_benchmark.jl`, 0.1 mm | 420 000 × 19 | 64 MB |

So per-step quantities — activation times, traces, movie frames — are observed during the
solve rather than saved. [`StateObserver`](@ref) calls a function with the live state every
`every` steps:

```julia
activation = fill(NaN, num_nodes(f))
record = StateObserver() do u, t
    φ = getvariable(u, f, :φₘ)
    for i in eachindex(activation)
        isnan(activation[i]) && φ[i] > 0.5 && (activation[i] = t)
    end
end
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, saveat = 10.0, callback = record)
```

Combine several with SciMLBase's `CallbackSet`; they run in order after each step.

A hand-written `DiscreteCallback` works too, but it needs two settings that `StateObserver`
already makes, and nothing warns when they are missing. Without `save_positions = (false, false)`
the default saves the full state twice every time the callback fires. Without
`derivative_discontinuity!(integrator, false)` the integrator assumes `u` changed and
resynchronizes its sub-problems on the next step.

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
  half owns scratch buffers and is stateful and single-threaded. For a threaded
  `EnsembleProblem`, build each trajectory's problem from its own `semidiscretize` call
  inside `prob_func`; the default `prob_func` hands every trajectory the same `f`.
- **Saving is opt-in and full-state.** See [Saving output](@ref); observe per-step
  quantities with a [`StateObserver`](@ref) rather than saving every step.
- **`solve` clashes with MatrixFreeOperators'.** Both packages export a `solve`; after
  `using FlashBang, MatrixFreeOperators`, write `FlashBang.solve`, or import
  MatrixFreeOperators by name.
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
