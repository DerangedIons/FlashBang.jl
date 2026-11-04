```@meta
CurrentModule = FlashBang
```

# The Solution Vector

State-blocked (structure of arrays), matching Thunderbolt's `StateBlockedLayout`:

```math
u = \left[\varphi_\mathrm{m}(1{:}N);\; s_1(1{:}N);\; s_2(1{:}N);\; \ldots;\; s_{M-1}(1{:}N)\right]
```

Every state is contiguous, so the diffusion half acts on the leading block `1:N` — no gather, no synchronizer object, and coalesced access on a GPU. Reach into it by name rather than by index.

## On a solution

Symbolic indexing works as it does for any SciML solution, with each name standing for a whole block of `N` values:

```julia
sol[:φₘ, end]                # copy of the voltage block at the last saved time
sol[:states, end]            # every non-voltage state, length N*(M-1)
sol[:CaMKt]                  # one named cell-model state, at every saved time
getvariable(sol, f, :φₘ)     # views instead of copies, one per saved time
```

The names are the cell model's state names, the model's `φ_symbol` (`:φₘ`, aliasing whatever the cell model calls its voltage), and its `state_symbol` (`:states`). The same names work on the problem (`prob[:φₘ]`) and on an integrator (`integrator[:φₘ]`).

## On a raw state vector

On an initial condition, an `integrator.u`, or a `sol.u[k]`, use [`getvariable`](@ref), [`setvariable!`](@ref), and [`variable_range`](@ref):

```julia
u₀ = create_initial_condition(f)       # every node at the cell model's default state
setvariable!(u₀, f, :φₘ) do x          # impose a profile
    x[1] < 1.0 ? 1.0 : 0.0
end
getvariable(u₀, f, :φₘ)                # view of the voltage block
getvariable(u₀, f, :states)            # view of every non-voltage state
```
