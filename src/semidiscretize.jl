"""
    DiffusionFunction(prepared, op, grid, stim, Cₘ, xs)

Diffusion half of the split: `∂ₜφₘ = ∇⋅(κ/(χCₘ))∇φₘ + Iₛₜᵢₘ/Cₘ` on the contiguous voltage
block of the state-blocked solution vector.

`prepared` is the MatrixFreeOperators `PreparedOperator` the right-hand side actually
applies; `op` and `grid` are kept alongside it because a prepared operator cannot be
rebuilt from itself — `Adapt` needs them to re-`prepare` on the target device, and tests
need them to inspect what was assembled.

!!! warning "One prepared operator per concurrent solve"
    `prepared` owns halo-padded scratch fields and is stateful and single-threaded. Two
    solves must not share a `DiffusionFunction`; call [`semidiscretize`](@ref) once per
    solve.

Not constructed directly; [`semidiscretize`](@ref) builds it.
"""
struct DiffusionFunction{P,O,G,S<:AbstractStimulationProtocol,T,X<:AbstractVector}
    prepared::P
    op::O
    grid::G
    stim::S
    Cₘ::T
    xs::X
end

function (f::DiffusionFunction)(du, u, p, t)
    mul!(du, f.prepared, u)
    apply_stimulus!(du, f.stim, f.xs, f.Cₘ, t)
    return nothing
end

"""
    apply_stimulus!(du, stim, xs, Cₘ, t, α = true) -> du

Add `α·Iₛₜᵢₘ(x, t)/Cₘ` to the voltage derivative at every node. A
[`NoStimulationProtocol`](@ref) dispatches to a no-op — the evaluation is removed, not
multiplied by zero — and a protocol with declared `nonzero_intervals` skips the broadcast
outside them.

`α` scales the source for callers that integrate it over a step rather than evaluate a
rate — [`CrankNicolson`](@ref) passes `Δt`. The default `true` is an exact multiplicative
identity, so the right-hand side is bit-for-bit what it was without it.
"""
apply_stimulus!(du, ::NoStimulationProtocol, xs, Cₘ, t, α=true) = du

function apply_stimulus!(du, stim::TransmembraneStimulationProtocol, xs, Cₘ, t, α=true)
    is_active(stim, t) || return du
    du .+= α .* stim.(xs, t) ./ Cₘ
    return du
end

function Adapt.adapt_structure(to, f::DiffusionFunction)
    op = Adapt.adapt(to, f.op)
    grid = Adapt.adapt(to, f.grid)
    xs = Adapt.adapt(to, f.xs)
    prepared = prepare(op, scalar_field(grid, eltype(eltype(xs))))
    return DiffusionFunction(prepared, op, grid, Adapt.adapt(to, f.stim), f.Cₘ, xs)
end

"""
    semidiscretize(split::ReactionDiffusionSplit, disc::FiniteDifferenceDiscretization,
                   grid::Union{CartesianGrid,BlockForest}) -> GenericSplitFunction

Turn a continuous [`ReactionDiffusionSplit`](@ref) into a right-hand side on `grid`,
ready for `OperatorSplittingProblem`.

`grid` is a uniform `CartesianGrid` or an adaptive `BlockForest`. On a forest the nodes are
the interior cells of every leaf, in the forest's flat (Morton block) order, and the result
describes the forest's *current* leaf set: after the forest changes, call [`regrid`](@ref),
which re-semidiscretizes and carries the solution across.

The result is a two-operator `GenericSplitFunction`:

| operator | dof range | right-hand side |
|---|---|---|
| 1 | `(k-1)N+1:kN` | [`DiffusionFunction`](@ref) — `∇⋅(κ/(χCₘ))∇φₘ + Iₛₜᵢₘ/Cₘ` |
| 2 | `1:N*M` | [`PointwiseODEFunction`](@ref) — the cell model at every node |

with `N` nodes, `M` states, and `k = CytoZoo.transmembrane_potential_index(ion)` — `1:N`
for the usual voltage-first model, but code generators that order states by dependency
(gotranx) put the voltage elsewhere. No synchronizer objects are needed: under the
state-blocked layout the voltage block the diffusion half acts on is a *contiguous* range
wherever it sits, so the two dof ranges overlap exactly where the physics couples them.

Only zero-flux (homogeneous `Neumann`) boundaries are supported in v0; anything else
raises an `ArgumentError` naming the offending face.

### Examples

```julia
using CytoZoo: FHNModel

grid = CartesianGrid(((0.0, 20.0),), (200,); bc = ((Neumann(), Neumann()),))
model = MonodomainModel(; κ = 1.0e-3, ion = FHNModel())
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
u₀ = create_initial_condition(f)
prob = OperatorSplittingProblem(f, u₀, (0.0, 100.0))
```

See also: [`create_initial_condition`](@ref), [`solution_size`](@ref).
"""
function semidiscretize(
    split::ReactionDiffusionSplit{<:MonodomainModel},
    disc::FiniteDifferenceDiscretization,
    grid::StructuredGrid{N},
) where {N}
    model = split.model
    _validate_zero_flux(grid)
    _validate_conductivity_dimension(model.κ, grid)

    T = _spacing_eltype(grid)
    op = diffusion_operator(model, grid)
    prepared = prepare(op, scalar_field(grid, T))

    xs = node_coordinates(grid)
    nnodes = length(xs)
    nstates = CytoZoo.num_states(model.ion)
    k = CytoZoo.transmembrane_potential_index(model.ion)
    1 <= k <= nstates || throw(
        ArgumentError(
            "the cell model reports transmembrane_potential_index = $k, outside its " *
            "$nstates states",
        ),
    )

    diffusion = DiffusionFunction(prepared, op, grid, model.stim, T(model.Cₘ), xs)
    reaction = PointwiseODEFunction(
        model.ion, xs, split.overrides, nnodes, nstates, model.φ_symbol, model.state_symbol
    )

    return GenericSplitFunction(
        (ODEFunction(diffusion), ODEFunction(reaction)),
        (_state_block(k, nnodes), 1:(nnodes * nstates)),
    )
end

"""
    diffusion_operator(model::MonodomainModel, grid::Union{CartesianGrid,BlockForest})
        -> AbstractOperator

The matrix-free operator `∇⋅(κ/(χCₘ))∇`, folded once at setup so the right-hand side
carries no coefficient arithmetic.

A scalar conductivity gives `κ_eff∇²`. An axis-aligned diagonal `κ` gives

```
κ_min∇² + Σ_d (κ_eff,d − κ_min)∂²/∂x_d²
```

which is exact for a diagonal tensor and cheaper than `N` separate second derivatives: the
Laplacian carries the isotropic part in one halo exchange and only the genuinely
anisotropic axes pay for a correction term. Axes equal to `κ_min` contribute nothing and
are dropped, so an isotropic tuple collapses back to the single-Laplacian form.
"""
function diffusion_operator(model::MonodomainModel, grid::StructuredGrid)
    return _diffusion_operator(model.κ ./ (model.χ * model.Cₘ), grid)
end

_diffusion_operator(κ_eff::Number, grid::StructuredGrid) = κ_eff * laplacian(grid)

# Built once per solve, so the type instability of accumulating an operator tree in a loop
# is paid at setup and never in `mul!` — every downstream call sees a concrete `prepared`.
function _diffusion_operator(κ_eff, grid::StructuredGrid{N}) where {N}
    κ_min = minimum(κ_eff)
    op = κ_min * laplacian(grid)
    for d in 1:N
        Δκ = κ_eff[d] - κ_min
        iszero(Δκ) && continue
        op = op + Δκ * derivative(grid, d; order=2)
    end
    return op
end

"""
    node_coordinates(grid::CartesianGrid{N}) -> AbstractVector{SVector{N}}
    node_coordinates(forest::BlockForest{N}) -> AbstractVector{SVector{N}}
    node_coordinates(f::GenericSplitFunction) -> AbstractVector{SVector{N}}

Cell-center coordinates of every interior cell, flattened in the same order
MatrixFreeOperators' `flatten` uses, on the grid's device. For a forest that is leaf by
leaf in Morton order, each leaf's cells in column-major order. Given a semidiscretization it
returns the very vector the right-hand sides use, so plotting `φₘ` against position needs
no second construction.

!!! note "Cell-centered, not vertex-centered"
    `n` cells across a length `L` put the first node at `Δx/2`, not at `0`. Comparisons
    against a vertex-centered reference (ArmyHeart) therefore differ at `O(Δx)` — a named
    numerical difference, not a bug.
"""
function node_coordinates(grid::CartesianGrid{N,T}) where {N,T}
    host = vec([cell_center(grid, I) for I in interior(grid)])
    backend = KernelAbstractions.get_backend(grid)
    backend isa KernelAbstractions.CPU && return host
    device = KernelAbstractions.allocate(backend, SVector{N,T}, length(host))
    copyto!(device, host)
    return device
end

# Host-side only: forests are regridded on the host, and FlashBang's forest path is CPU-only.
function node_coordinates(forest::BlockForest{N,T}) where {N,T}
    per_leaf = map(1:nleaves(forest)) do i
        leaf = leaf_grid(forest, i)
        return vec([cell_center(leaf, I) for I in interior(leaf)])
    end
    return reduce(vcat, per_leaf)
end

_spacing_eltype(grid::CartesianGrid) = eltype(spacing(grid))
_spacing_eltype(::BlockForest{N,T}) where {N,T} = T

# MatrixFreeOperators has no `boundary_conditions` method for a forest; the physical faces
# are the base grid's, stored on the forest.
_boundary_conditions(grid::CartesianGrid) = boundary_conditions(grid)
_boundary_conditions(forest::BlockForest) = forest.bc

function _validate_zero_flux(grid::StructuredGrid{N}) where {N}
    bcs = _boundary_conditions(grid)
    for d in 1:N, (side, bc) in zip(("low", "high"), bcs[d])
        if !(bc isa Neumann) || !iszero(bc.flux)
            throw(
                ArgumentError(
                    "v0 supports zero-flux boundaries only; dimension $d, $side face is " *
                    "$(bc). Build the grid with " *
                    "`bc = ntuple(_ -> (Neumann(), Neumann()), $N)`.",
                ),
            )
        end
    end
    return nothing
end

_validate_conductivity_dimension(::Number, ::StructuredGrid) = nothing

function _validate_conductivity_dimension(κ, grid::StructuredGrid{N}) where {N}
    length(κ) == N || throw(
        ArgumentError(
            "per-axis conductivity has $(length(κ)) entries but the grid is " *
            "$N-dimensional; pass one κ per dimension or a scalar for isotropic tissue",
        ),
    )
    return nothing
end
