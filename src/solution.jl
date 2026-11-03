# Named access to the state-blocked solution vector. The layout is
#
#     u = [φₘ(1:N); s₁(1:N); … ; s_{M-1}(1:N)]
#
# so every state is a contiguous block and `getvariable` is a view, never a gather. The
# voltage is drawn first, but it is block `CytoZoo.transmembrane_potential_index(ion)`
# wherever that is; everything below goes through `_state_position` rather than assuming 1.

# The split function's two operators, in the order `semidiscretize` builds them. Reaching
# through `ODEFunction`'s `.f` is the price of returning a plain `GenericSplitFunction`
# (which is what makes the pipeline congruent with Thunderbolt) instead of a wrapper type.
_diffusion_function(f::GenericSplitFunction) = f.functions[1].f
_reaction_function(f::GenericSplitFunction) = f.functions[2].f

_coordinate_type(f::GenericSplitFunction) = eltype(eltype(_reaction_function(f).xs))

node_coordinates(f::GenericSplitFunction) = _reaction_function(f).xs

"""
    num_nodes(f::GenericSplitFunction) -> Int

Number of spatial nodes `N` in a semidiscretized problem.
"""
num_nodes(f::GenericSplitFunction) = _reaction_function(f).nnodes

# Deliberately *not* a method on `CytoZoo.num_states`: that function and
# `GenericSplitFunction` are both owned elsewhere, so extending it here would be type piracy.
_num_states(f::GenericSplitFunction) = _reaction_function(f).nstates

"""
    solution_size(f::GenericSplitFunction) -> Int

Length of the solution vector: `N * M` for `N` nodes and `M` cell-model states per node.
"""
solution_size(f::GenericSplitFunction) = num_nodes(f) * _num_states(f)

"""
    create_initial_condition(f::GenericSplitFunction, [T]) -> AbstractVector{T}

Allocate the solution vector and fill every node with the cell model's
`default_initial_state`, on the same device the semidiscretization lives on.

`T` defaults to the grid's coordinate element type (`Float64` for a `Float64` grid), so
the initial condition matches the operator rather than silently mixing precisions.

This is the *spatially uniform* starting point. Impose structure on top of it with
[`setvariable!`](@ref) — an initial voltage profile, a heterogeneous gating state, and so
on.

!!! warning "Not necessarily a resting state"
    `default_initial_state` is whatever the cell model ships. For `ToRORd` it is not
    pre-paced (`dφₘ/dt ≈ +240 mV/ms` at `t = 0`); pace a single cell first and
    `setvariable!` the settled values if the opening transient matters.

### Examples

```julia
u₀ = create_initial_condition(f)
setvariable!(u₀, f, :φₘ) do x
    x[1] < 1.0 ? 1.0 : 0.0
end
```
"""
function create_initial_condition(f::GenericSplitFunction)
    return create_initial_condition(f, _coordinate_type(f))
end

function create_initial_condition(f::GenericSplitFunction, ::Type{T}) where {T}
    reaction = _reaction_function(f)
    N, M = reaction.nnodes, reaction.nstates
    backend = KernelAbstractions.get_backend(reaction.xs)
    u = KernelAbstractions.allocate(backend, T, N * M)
    u₀ = CytoZoo.default_initial_state(reaction.ion)
    for k in 1:M
        fill!(view(u, _state_block(k, N)), T(u₀[k]))
    end
    return u
end

"""
    getvariable(u, f::GenericSplitFunction, name::Symbol) -> SubArray

View of one named block of the solution vector.

`name` is a cell-model state name, the model's `φ_symbol` (`:φₘ` by default, aliasing
whichever state the cell model reports as the transmembrane potential), or its
`state_symbol` (`:states` by default) for the whole non-voltage block. The last is only
contiguous — and therefore only available — when the voltage is the first state; see
[`variable_range`](@ref).

Being a view, writing through it writes the solution.

### Examples

```julia
φ = getvariable(u, f, :φₘ)      # length N
s = getvariable(u, f, :states)  # length N*(M-1), all non-voltage states
w = getvariable(u, f, :CaMKt)   # length N, one named state
```
"""
getvariable(u, f::GenericSplitFunction, name::Symbol) = view(u, variable_range(f, name))

"""
    setvariable!(fun, u, f::GenericSplitFunction, name::Symbol) -> u
    setvariable!(u, f::GenericSplitFunction, name::Symbol, value::Number) -> u

Fill one named state block by evaluating `fun(x)` at every node's cell-center coordinate,
or with a constant `value`.

Reads naturally with `do`-block syntax, since the function is the first argument.

`name` must name a single state (a cell-model state name or the model's `φ_symbol`); the
whole-state-block symbol is rejected, because one function of position cannot say what to
put in each of several states.

### Examples

```julia
setvariable!(u, f, :φₘ) do x
    cos(π * x[1] / L)
end
setvariable!(u, f, :φₘ, 0.0)
```
"""
function setvariable!(fun, u, f::GenericSplitFunction, name::Symbol)
    reaction = _reaction_function(f)
    k = _state_position(reaction, name)
    block = view(u, _state_block(k, reaction.nnodes))
    block .= fun.(reaction.xs)
    return u
end

function setvariable!(u, f::GenericSplitFunction, name::Symbol, value::Number)
    reaction = _reaction_function(f)
    k = _state_position(reaction, name)
    fill!(view(u, _state_block(k, reaction.nnodes)), value)
    return u
end

"""
    variable_range(f::GenericSplitFunction, name::Symbol) -> UnitRange{Int}

Index range of a named block in the solution vector — the raw form of
[`getvariable`](@ref), useful for building views of an integrator's `u` without holding
one.

The whole non-voltage block (`state_symbol`, `:states` by default) is `(N+1):(N*M)` when
the voltage is the first state. When the cell model reports another
`transmembrane_potential_index` the non-voltage states straddle the voltage block, so there
is no single range to return and an `ArgumentError` is thrown; address the states by name
instead.
"""
function variable_range(f::GenericSplitFunction, name::Symbol)
    reaction = _reaction_function(f)
    N, M = reaction.nnodes, reaction.nstates
    if name === reaction.state_symbol
        k = CytoZoo.transmembrane_potential_index(reaction.ion)
        k == 1 || throw(
            ArgumentError(
                ":$name is not a contiguous range: the voltage is state $k of $M, so the " *
                "non-voltage states sit on both sides of it; address them by name",
            ),
        )
        return (N + 1):(N * M)
    end
    return _state_block(_state_position(reaction, name), N)
end

@inline _state_block(k::Integer, nnodes::Integer) = ((k - 1) * nnodes + 1):(k * nnodes)

function _state_position(reaction::PointwiseODEFunction, name::Symbol)
    name === reaction.φ_symbol && return CytoZoo.transmembrane_potential_index(reaction.ion)
    names = CytoZoo.state_names(reaction.ion)
    # `findfirst` rather than `CytoZoo.state_index`, which throws a `KeyError` on some
    # models instead of returning `nothing`.
    k = findfirst(==(name), names)
    k === nothing && throw(
        ArgumentError(
            "no state named :$name; the cell model has $(names), and :$(reaction.φ_symbol) " *
            "aliases the transmembrane potential (:$(reaction.state_symbol) selects the " *
            "whole non-voltage block, which `setvariable!` does not accept)",
        ),
    )
    return k
end
