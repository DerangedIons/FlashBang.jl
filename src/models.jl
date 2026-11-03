"""
    AbstractEPModel

Supertype for cardiac electrophysiology tissue models. A model is a *declaration* of the
continuous problem — coefficients, cell model, stimulation — carrying no discretization.
[`semidiscretize`](@ref) turns one into a right-hand side on a given grid.
"""
abstract type AbstractEPModel end

"""
    MonodomainModel(; κ, ion, χ = 1, Cₘ = 1, stim = NoStimulationProtocol(),
                      φ_symbol = :φₘ, state_symbol = :states)

The monodomain equation

```
χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ(φₘ, s, t) + Iₛₜᵢₘ(x, t))
                 ∂ₜs = f(φₘ, s, t)
```

for transmembrane potential `φₘ` and cell-model states `s`.

# Keyword Arguments
- `κ`: conductivity. A `Number` is isotropic; an `NTuple{N}` or `SVector{N}` is
  axis-aligned diagonal anisotropy, one entry per grid dimension (fibres along an axis).
  Spatially varying conductivity fields are not supported in v0.
- `ion`: any `CytoZoo.AbstractCellModel`. Supplies `Iᵢₒₙ` and the state kinetics; it *is*
  the reaction half of the split.
- `χ`: surface-to-volume ratio.
- `Cₘ`: membrane capacitance per unit area.
- `stim`: an [`AbstractStimulationProtocol`](@ref). Applied in the diffusion half, as a
  source term, uniformly in every dimension.
- `φ_symbol`, `state_symbol`: names the solution helpers ([`getvariable`](@ref),
  [`setvariable!`](@ref)) and symbolic indexing (`sol[name]`, `integrator[name]`) accept
  for the voltage block and for the whole non-voltage state block. `state_symbol` is deliberately plural: a cell model may itself name a state `:s`
  (CytoZoo's `FHNModel` does), and a singular default would shadow it.

The constructor is **keyword-only on purpose.** Thunderbolt's tutorial calls
`MonodomainModel(Cₘ, χ, …)` while its fields are ordered `(χ, Cₘ, …)`; the two are
dimensionally interchangeable in the equation above, so a swapped pair is silent. Naming
them removes the trap.

# Units

Nothing here fixes a unit system — `κ`, `χ`, `Cₘ`, the stimulus, and the cell model must
simply agree. What the code does fix is that `κ` enters as the diffusivity `κ/(χCₘ)` and
the stimulus as `Iₛₜᵢₘ/Cₘ`.

!!! warning "Use `Float64`"
    The types are generic and `Float32` runs, but the ArmyHeart reference measured
    non-finite results across 58.6% of the ToRORd parameter space in `Float32`. Treat
    single precision as a performance experiment, not a production setting.

### Examples

```julia
using CytoZoo: FHNModel

model = MonodomainModel(; κ = 1.0e-3, ion = FHNModel())

# fibres along x, three-dimensional anisotropy
model = MonodomainModel(;
    κ = (0.133, 0.0176, 0.0176),
    χ = 140.0,
    Cₘ = 0.01,
    ion = FHNModel(),
)
```

See also: [`ReactionDiffusionSplit`](@ref), [`semidiscretize`](@ref).
"""
struct MonodomainModel{Tχ,TC,Tκ,TS<:AbstractStimulationProtocol,TI} <: AbstractEPModel
    χ::Tχ
    Cₘ::TC
    κ::Tκ
    stim::TS
    ion::TI
    φ_symbol::Symbol
    state_symbol::Symbol
end

function MonodomainModel(;
    κ,
    ion,
    χ=1,
    Cₘ=1,
    stim::AbstractStimulationProtocol=NoStimulationProtocol(),
    φ_symbol::Symbol=:φₘ,
    state_symbol::Symbol=:states,
)
    _validate_conductivity(κ)
    χ > zero(χ) || throw(ArgumentError("χ must be positive, got $χ"))
    Cₘ > zero(Cₘ) || throw(ArgumentError("Cₘ must be positive, got $Cₘ"))
    φ_symbol === state_symbol &&
        throw(ArgumentError("φ_symbol and state_symbol must differ, both are :$φ_symbol"))
    _warn_on_active_cell_stimulus(ion)
    return MonodomainModel(χ, Cₘ, κ, stim, ion, φ_symbol, state_symbol)
end

function _validate_conductivity(κ::Number)
    κ > zero(κ) || throw(ArgumentError("conductivity κ must be positive, got $κ"))
    return nothing
end

function _validate_conductivity(κ)
    isempty(κ) && throw(ArgumentError("per-axis conductivity κ must not be empty"))
    all(>(zero(eltype(κ))), κ) ||
        throw(ArgumentError("every per-axis conductivity must be positive, got $κ"))
    return nothing
end

# Best-effort guard against double stimulation. The tissue model owns the stimulus (see
# `AbstractStimulationProtocol`), so a cell model still carrying a live one would add a
# second — with the opposite sign, which makes the result confusing rather than merely
# large. Only `CytoZoo.Stimulus` can be inspected cheaply; a `FunctionStimulus` or a custom
# subtype is left alone.
function _warn_on_active_cell_stimulus(ion)
    hasproperty(ion, :stim) || return nothing
    s = ion.stim
    if s isa CytoZoo.Stimulus && !iszero(s.amplitude)
        @warn """
        The cell model carries a nonzero stimulus (amplitude $(s.amplitude)) *and* the \
        tissue model applies its own in the diffusion half. These add, and they use \
        opposite sign conventions. Pass a zero-amplitude stimulus to the cell model — \
        e.g. `Stimulus(; amplitude = 0)` — and stimulate through the tissue model's \
        `stim` keyword.
        """ maxlog = 1
    end
    return nothing
end

"""
    ReactionDiffusionSplit(model)
    ReactionDiffusionSplit(model, overrides)

Annotate `model` for a reaction-diffusion operator split: the ionic kinetics are advanced
pointwise, the diffusion operator globally. This is a declaration only — no work happens
until [`semidiscretize`](@ref).

`overrides` is the `NamedTuple` of spatial parameter overrides handed to the cell model
through a `CytoZoo.SpatialContext` at every node. This is how spatial heterogeneity enters:
cell type, pH, hypoxia, or any other parameter the cell model resolves spatially. Values
may be numbers, callables `(x, t) -> v`, or isbits `CytoZoo.SpatialFunction` functors
(`Constant`, `SpatialStep`, `SpatialGradient`) — only the last kind is GPU-safe.

With no overrides the reaction half calls the cell model with `p = nothing`, which is the
fast non-spatial dispatch every CytoZoo model compiles down to.

### Examples

```julia
using CytoZoo: SpatialStep

split = ReactionDiffusionSplit(model)

# endocardial on the left half of x, epicardial on the right
split = ReactionDiffusionSplit(model, (celltype = SpatialStep(1, 0.5, 0.0, 1.0),))
```
"""
struct ReactionDiffusionSplit{M<:AbstractEPModel,O}
    model::M
    overrides::O
end

ReactionDiffusionSplit(model::AbstractEPModel) = ReactionDiffusionSplit(model, nothing)
