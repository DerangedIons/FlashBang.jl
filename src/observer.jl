"""
    StateObserver(observe; every = 1) -> DiscreteCallback

A callback that calls `observe(u, t)` after every `every`-th accepted step of a `solve`,
with the live state vector `u` and the time `t` it has just reached. Use it for per-step
quantities too large or too frequent to save — activation times, voltage traces, movie
frames — and read the state by name inside it with [`getvariable`](@ref).

`every` counts steps, not time. `tstops` can make the integrator take an extra, near-zero
step to land exactly on a stop time; that step counts, and shifts every later firing by
one step. Leave `tstops` out when an observer must stay on a fixed time grid.

`u` is the integrator's own buffer: read from it, copy what you keep (`collect(φ)`), and
do not write to it.

Two flags are preset, because the obvious hand-written `DiscreteCallback` gets both wrong
on the operator-splitting integrator, silently:

- `save_positions = (false, false)`. The default `(true, true)` saves the full state
  before and after every firing, on top of `saveat` — a callback that fires every step
  turns a run with six saves into one with two thousand.
- `derivative_discontinuity!(integrator, false)` after `observe`. Without it the
  integrator assumes the callback may have changed `u` and resynchronizes its sub-problems
  on the next step, which copies the full state twice per step.

### Examples

```julia
activation = fill(NaN, num_nodes(f))
record = StateObserver() do u, t
    φ = getvariable(u, f, :φₘ)
    for i in eachindex(activation)
        isnan(activation[i]) && φ[i] > 0.5 && (activation[i] = t)
    end
end
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt, saveat = 1.0, callback = record)
```
"""
function StateObserver(observe; every::Integer=1)
    every ≥ 1 || throw(ArgumentError("`every` must be a positive step count, got $every"))
    return DiscreteCallback(
        _EveryNSteps(Int(every)), _Observe(observe); save_positions=(false, false)
    )
end

# `naccept` is incremented before callbacks run, so at condition time it counts accepted
# steps including the one just completed: this fires after steps `every`, `2every`, ….
# Not `iter`, which also counts rejected attempts under an adaptive splitting algorithm.
struct _EveryNSteps
    every::Int
end
(c::_EveryNSteps)(u, t, integrator) = integrator.stats.naccept % c.every == 0

struct _Observe{F}
    observe::F
end
function (a::_Observe)(integrator)
    a.observe(integrator.u, integrator.t)
    SciMLBase.derivative_discontinuity!(integrator, false)
    return nothing
end
