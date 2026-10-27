# Inner algorithms for the two halves of the split. Each is an ordinary OrdinaryDiffEq
# algorithm — `LieTrotterGodunov((CrankNicolson(), RushLarsen()))` builds one leaf
# `ODEIntegrator` per half exactly as it does for `Euler()` — but it does not treat its
# half as a black-box right-hand side. Both unwrap the FlashBang functor the split handed
# them and use structure an RHS cannot express: the cell model's own exponential step, and
# the linearity of diffusion.
#
# Both are fixed-step and non-FSAL, after OrdinaryDiffEqFunctionMap. `loopheader!` copies
# `u → uprev` before every `perform_step!`, so `uprev` is the start-of-step state and `u`
# the destination: a free ping-pong pair for an out-of-place step.

"""
    AlgorithmStats

Wall-clock and work counters for one half of the split, accumulated over every step the
algorithm has taken. Reach it with [`algorithm_stats`](@ref).

# Fields
- `elapsed_ns::Int`: wall time inside `perform_step!`, bracketed by device synchronizations
  so asynchronous GPU work is charged to the step that queued it.
- `steps::Int`: steps taken.
- `iterations::Int`: inner linear-solver iterations (CG for [`CrankNicolson`](@ref)); `0`
  for [`RushLarsen`](@ref).

The record is mutable and live — the integrator keeps writing into it. The first step
includes JIT compilation (for a large generated cell model, seconds of it), so a driver
reporting time per step should zero the fields after a warm-up step:
`s.elapsed_ns = 0; s.steps = 0; s.iterations = 0`.
"""
mutable struct AlgorithmStats
    elapsed_ns::Int
    steps::Int
    iterations::Int
end

AlgorithmStats() = AlgorithmStats(0, 0, 0)

"""
    algorithm_stats(integrator) -> AlgorithmStats, Tuple, or nothing

The [`AlgorithmStats`](@ref) of a FlashBang inner algorithm.

Given the top-level operator-splitting integrator, returns one entry per operator in split
order — for `LieTrotterGodunov((CrankNicolson(), RushLarsen()))` that is
`(diffusion_stats, reaction_stats)`. A nested split gives a nested tuple, and an operator
advanced by an algorithm that keeps no stats (`Euler()`, say) gives `nothing`. Given a
single leaf integrator (one entry of `integrator.child_subintegrators`), returns its record
alone.

### Examples

```julia
integrator = init(prob, LieTrotterGodunov((CrankNicolson(), RushLarsen())); dt)
step!(integrator, dt, true)
(pde, ode) = algorithm_stats(integrator)
ode.elapsed_ns / ode.steps / 1e6          # reaction ms per step
pde.iterations / pde.steps                # mean CG iterations per diffusion step
```
"""
function algorithm_stats(integrator::OrdinaryDiffEqOperatorSplitting.AnySplitIntegrator)
    return map(algorithm_stats, integrator.child_subintegrators)
end
function algorithm_stats(integrator::OrdinaryDiffEqCore.ODEIntegrator)
    return algorithm_stats(integrator.cache)
end
algorithm_stats(::OrdinaryDiffEqCore.OrdinaryDiffEqCache) = nothing

# The integrator hands `alg_cache` its function wrapped in an `ODEFunction` and possibly a
# `FunctionWrappersWrapper` on top; both algorithms need the FlashBang functor underneath.
_split_operator(f::SciMLBase.AbstractSciMLFunction) = SciMLBase.unwrapped_f(f.f)
_split_operator(f) = SciMLBase.unwrapped_f(f)

# Bracket a step with device synchronizations, so the time charged is this half's alone:
# work an earlier stage queued is drained before the clock starts, and work this step
# queued is drained before it stops. A no-op on the CPU.
function _timed!(step, stats::AlgorithmStats, backend)
    _synchronize(backend)
    t₀ = time_ns()
    step()
    _synchronize(backend)
    stats.elapsed_ns += Int(time_ns() - t₀)
    stats.steps += 1
    return nothing
end

_synchronize(::KernelAbstractions.CPU) = nothing
# `invokelatest` for the same reason as `_react!`: the method comes from a GPU package
# that need not be loaded.
_synchronize(backend) = Base.invokelatest(KernelAbstractions.synchronize, backend)

#--------------------------------------------------------------------------------# RushLarsen

"""
    RushLarsen()

Reaction-half algorithm that advances every node with the cell model's own
`CytoZoo.rush_larsen_step!` instead of integrating its right-hand side.

A Rush–Larsen step (or a generalized or blocked variant of it) treats each gating variable
as linear in itself over the step and integrates it with the exact exponential — the same
reason an integrating factor removes a stiff linear term. The stiffness limit of explicit
Euler on fast gates disappears, so the reaction half can take steps an order of magnitude
longer. The step is whatever the cell model implements (CytoZoo's `ToRORd`, or gotranx
`ode2cytozoo --scheme generalized_rush_larsen|block_rush_larsen` output); FlashBang only
calls it, one node per thread on the CPU and one node per work-item on a GPU.

The cell model must report `CytoZoo.has_rush_larsen(ion) == true`; `init` throws an
`ArgumentError` otherwise. Fixed-step only.

### Examples

```julia
alg = LieTrotterGodunov((CrankNicolson(), RushLarsen()))
integrator = init(OperatorSplittingProblem(f, u₀, (0.0, 100.0)), alg; dt = 0.05)
```

See also: [`CrankNicolson`](@ref), [`algorithm_stats`](@ref).
"""
struct RushLarsen <: OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm end

struct RushLarsenCache{uType,F<:PointwiseODEFunction} <:
       OrdinaryDiffEqCore.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    reaction::F
    stats::AlgorithmStats
end

OrdinaryDiffEqCore.isfsal(::RushLarsen) = false
OrdinaryDiffEqCore.alg_order(::RushLarsen) = 1
OrdinaryDiffEqCore.get_fsalfirstlast(::RushLarsenCache, u) = (nothing, nothing)
SciMLBase.full_cache(c::RushLarsenCache) = (c.u, c.uprev)
algorithm_stats(cache::RushLarsenCache) = cache.stats

function OrdinaryDiffEqCore.alg_cache(
    ::RushLarsen,
    u,
    rate_prototype,
    ::Type{uEltypeNoUnits},
    ::Type{uBottomEltypeNoUnits},
    ::Type{tTypeNoUnits},
    uprev,
    uprev2,
    f,
    t,
    dt,
    reltol,
    p,
    calck,
    ::Val{true},
    verbose,
) where {uEltypeNoUnits,uBottomEltypeNoUnits,tTypeNoUnits}
    reaction = _split_operator(f)
    reaction isa PointwiseODEFunction || throw(
        ArgumentError(
            "RushLarsen() advances FlashBang's reaction half; it was given " *
            "$(typeof(reaction)). Put it in the reaction slot of the splitting tuple.",
        ),
    )
    CytoZoo.has_rush_larsen(reaction.ion) || throw(
        ArgumentError(
            "RushLarsen() needs a cell model with a Rush–Larsen step, but " *
            "CytoZoo.has_rush_larsen returns false for $(typeof(reaction.ion)). Use an " *
            "RHS integrator such as Euler() instead, or generate the model with a " *
            "Rush–Larsen scheme.",
        ),
    )
    return RushLarsenCache(u, uprev, reaction, AlgorithmStats())
end

function OrdinaryDiffEqCore.initialize!(integrator, ::RushLarsenCache)
    integrator.kshortsize = 0
    resize!(integrator.k, integrator.kshortsize)
    return nothing
end

function OrdinaryDiffEqCore.perform_step!(
    integrator, cache::RushLarsenCache, repeat_step=false
)
    (; u, uprev, t, dt) = integrator
    backend = KernelAbstractions.get_backend(u)
    _timed!(cache.stats, backend) do
        return _rush_larsen!(u, uprev, cache.reaction, t, dt, backend)
    end
    return nothing
end

function _rush_larsen!(u, uprev, f::PointwiseODEFunction, t, dt, ::KernelAbstractions.CPU)
    N, M = f.nnodes, f.nstates
    # Each node reads its own stride of `uprev` and writes its own stride of `u`.
    Threads.@threads for i in 1:N
        CytoZoo.rush_larsen_step!(
            _node_slice(u, i, N, M),
            _node_slice(uprev, i, N, M),
            _node_context(f.xs, f.overrides, i),
            t,
            dt,
            f.ion,
        )
    end
    return nothing
end

# Synchronization is `_timed!`'s job, so unlike `_react!` this only launches.
function _rush_larsen!(u, uprev, f::PointwiseODEFunction, t, dt, backend)
    kernel = _rush_larsen_kernel!(backend)
    Base.invokelatest(
        kernel,
        u,
        uprev,
        f.ion,
        f.xs,
        f.overrides,
        f.nnodes,
        f.nstates,
        t,
        dt;
        ndrange=f.nnodes,
    )
    return nothing
end

@kernel function _rush_larsen_kernel!(u, uprev, ion, xs, overrides, nnodes, nstates, t, dt)
    i = @index(Global, Linear)
    CytoZoo.rush_larsen_step!(
        _node_slice(u, i, nnodes, nstates),
        _node_slice(uprev, i, nnodes, nstates),
        _node_context(xs, overrides, i),
        t,
        dt,
        ion,
    )
end
