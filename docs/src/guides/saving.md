```@meta
CurrentModule = FlashBang
```

# Saving Output

Saving works differently from OrdinaryDiffEq's defaults, in ways that matter for a field problem:

- **It is opt-in.** Without `saveat`, only the first and last states are kept.
- **Every save is the whole state** — every cell-model state at every node. `save_idxs` is not supported: it does not shrink saves.
- **A `saveat` time that falls inside a step is linearly interpolated.** Pass the same times as `tstops` to make a step land on each one. Accumulated rounding in `t` can then force an extra, near-zero step to reach a stop, which shifts a step-counted [`StateObserver`](@ref) off its time grid. When the save times are multiples of `dt`, leave `tstops` out.

What one save costs, in `Float64`:

| Problem | Nodes × states | One save |
|---|---|---|
| `cable_1d.jl` | 400 × 2 | 6.4 KB |
| `spiral_2d.jl` | 256² × 2 | 1.05 MB |
| `niederer_benchmark.jl`, 0.1 mm | 420 000 × 19 | 64 MB |

## Observing instead of saving

Per-step quantities — activation times, traces, movie frames — are observed during the solve rather than saved. [`StateObserver`](@ref) calls a function with the live state every `every` steps:

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

Combine several with SciMLBase's `CallbackSet`; they run in order after each step. On a GPU the loop above has to become a broadcast; see [Running on a GPU](gpu.md#Observers-on-the-device).

## Writing your own callback

A hand-written `DiscreteCallback` works too, but it needs two settings that `StateObserver` already makes, and nothing warns when they are missing. Without `save_positions = (false, false)` the default saves the full state twice every time the callback fires. Without `derivative_discontinuity!(integrator, false)` the integrator assumes `u` changed and resynchronizes its sub-problems on the next step.
