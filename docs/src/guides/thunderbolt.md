```@meta
CurrentModule = FlashBang
```

# Thunderbolt compatibility

FlashBang and [Thunderbolt.jl](https://github.com/JuliaHealth/Thunderbolt.jl) solve the same monodomain problem on different geometry: structured grids here, unstructured finite-element meshes there. FlashBang uses Thunderbolt's names for everything that describes the problem, so a model written for one reads the same in the other. Only the geometry and discretization lines change.

The difference is at the end. Thunderbolt's tutorials step the integrator by hand. FlashBang's primary API is SciML's `solve` (see [Getting Started](../getting_started.md)). The hand-driven form is still fully supported, and this page covers it.

## Shared vocabulary

| Name | Role | Same in Thunderbolt |
|---|---|---|
| [`MonodomainModel`](@ref) | the continuous problem | yes |
| [`TransmembraneStimulationProtocol`](@ref) | the applied current | the name is shared, but in Thunderbolt it is the abstract type and the concrete analytic protocol is `AnalyticalTransmembraneStimulationProtocol` |
| [`ReactionDiffusionSplit`](@ref) | how to split it | yes |
| [`semidiscretize`](@ref) | builds the split function | yes |
| `OperatorSplittingProblem`, `LieTrotterGodunov` | the split problem and algorithm (OrdinaryDiffEqOperatorSplitting) | yes |
| [`create_initial_condition`](@ref) | every node at the cell model's default state | yes |
| [`getvariable`](@ref), [`setvariable!`](@ref) | named access to the state vector | yes |
| `CartesianGrid`, [`FiniteDifferenceDiscretization`](@ref) | geometry and discretization | no: these are the lines that change |

The solution vector also follows Thunderbolt's state-blocked layout (Thunderbolt calls it `StateBlockedLayout`), so a block of `u` means the same thing in both.

## Driving the integrator directly

```julia
prob = OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 400.0))
dt = 0.005
integrator = init(prob, LieTrotterGodunov((Euler(), Euler())); dt)

for _ in 1:round(Int, 400.0 / dt)
    step!(integrator, dt, true)
    φ = getvariable(integrator.u, f, :φₘ)     # a view, not a copy
    # ... inspect or modify the state between steps
end
```

Count steps rather than looping `while integrator.t < 400.0`. Time accumulates rounding error, so `t` lands a hair below 400 after the last step and the `while` form takes one step too many. `solve` handles this itself and lands exactly on the end of `tspan`. That is also why its final state can differ slightly from a fixed-count loop's when `dt` is not exactly representable in binary: `solve` takes one extra, near-zero step to land on the end time.

`solve!(integrator)` runs the remaining steps and returns the same solution object `solve` would, with the same `saveat` and `callback` behaviour.

## When to drive it by hand

`solve` with a [`StateObserver`](@ref) covers per-step reading: activation times, traces, movie frames. Step by hand when the loop has to do more than read:

- **Changing state mid-run** with [`setvariable!`](@ref) on `integrator.u` (an S2 written as a state reset rather than a stimulus current, for example). A callback can do this too, but then it must not call `derivative_discontinuity!(integrator, false)`, because the state really did change.
- **Interactive or conditional stepping**, such as advancing until a condition holds and then deciding what to do next.
- **Porting a Thunderbolt script** line for line before moving it to `solve`.

## Names on an integrator

Symbolic indexing works on the integrator as well as on the solution. `integrator[:φₘ]` returns the voltage block. It copies on every call, though: at the 0.1 mm Niederer resolution that is 3.4 MB per call, so inside a stepping loop use `getvariable(integrator.u, f, :φₘ)`, which is a view.
