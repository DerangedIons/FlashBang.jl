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

#--------------------------------------------------------------------------------# CrankNicolson

"""
    CrankNicolson(; reltol = 1e-10, maxiter = 200)

Diffusion-half algorithm: the trapezoidal rule on `∂ₜφ = Pφ + Iₛₜᵢₘ/Cₘ`, solved matrix-free
with conjugate gradients.

```
(I − (Δt/2)P) φⁿ⁺¹ = (I + (Δt/2)P) φⁿ + Δt·Iₛₜᵢₘ(x, tⁿ + Δt/2)/Cₘ
```

Explicit Euler on diffusion is stable only while `Δt ≤ 1/(2Σ_d D_d/h_d²)` — about 0.04 ms
on the Niederer slab at 0.1 mm — because the fastest grid mode decays at a rate
`4Σ_d D_d/h_d²` and Euler overshoots it. Crank–Nicolson is unconditionally stable and second order, at the price of
one linear solve per step. That solve is cheap here: `P` is the zero-flux diffusion
operator, symmetric negative semidefinite, so `I − (Δt/2)P` is symmetric positive definite
with condition number `1 + Δt·λ_max(−P)/2` — a few units at cardiac `Δt` and `h`. Plain
CG, warm-started from `φⁿ`, converges in tens of iterations without a preconditioner, and
needs nothing but the prepared operator's `mul!`, so it runs unchanged on a GPU.

The stimulus is sampled once, at the step midpoint, which keeps the scheme second order
for a smooth source (the same choice as fenicsx-beat). A pulse shorter than `Δt` can be
missed entirely; resolve it with `Δt` or widen it.

# Keyword Arguments
- `reltol`: CG stops once `‖b − Aφ‖ ≤ reltol·‖b‖` — relative to the right-hand side, as
  PETSc's default test, rather than to the initial residual: with a warm start the initial
  residual is already tiny wherever the tissue is at rest, and asking for a further factor
  of `reltol` there would chase round-off for `maxiter` iterations.
- `maxiter`: CG iteration cap. A step that hits it without converging is failed (the
  splitting integrator stops with a non-success retcode), not silently accepted.

Fixed-step only. Iteration counts are kept in [`algorithm_stats`](@ref).

### Examples

```julia
alg = LieTrotterGodunov((CrankNicolson(; reltol = 1e-10), RushLarsen()))
```
"""
struct CrankNicolson{T} <: OrdinaryDiffEqCore.OrdinaryDiffEqAlgorithm
    reltol::T
    maxiter::Int
end

function CrankNicolson(; reltol=1.0e-10, maxiter::Integer=200)
    reltol > zero(reltol) || throw(ArgumentError("reltol must be positive, got $reltol"))
    maxiter > 0 || throw(ArgumentError("maxiter must be positive, got $maxiter"))
    return CrankNicolson(reltol, Int(maxiter))
end

# `y = x − (Δt/2)Px` — all Krylov needs of an operator is `mul!`, `size` and `eltype`.
# Rebuilt every step from the integrator's current `dt`, which costs nothing (it only
# wraps a reference) and means a changed `dt` can never leave a stale operator behind.
struct _CNOperator{P,T}
    prepared::P
    halfdt::T
end

Base.size(A::_CNOperator) = size(A.prepared)
Base.size(A::_CNOperator, d::Integer) = size(A.prepared, d)
Base.eltype(A::_CNOperator) = eltype(A.prepared)

function LinearAlgebra.mul!(y::AbstractVector, A::_CNOperator, x::AbstractVector)
    copyto!(y, x)
    mul!(y, A.prepared, x, -A.halfdt, true)
    return y
end

struct CrankNicolsonCache{uType,F<:DiffusionFunction,W,T} <:
       OrdinaryDiffEqCore.OrdinaryDiffEqMutableCache
    u::uType
    uprev::uType
    rhs::uType
    diffusion::F
    workspace::W
    reltol::T
    maxiter::Int
    stats::AlgorithmStats
end

OrdinaryDiffEqCore.isfsal(::CrankNicolson) = false
OrdinaryDiffEqCore.alg_order(::CrankNicolson) = 2
OrdinaryDiffEqCore.get_fsalfirstlast(::CrankNicolsonCache, u) = (nothing, nothing)
SciMLBase.full_cache(c::CrankNicolsonCache) = (c.u, c.uprev, c.rhs)
algorithm_stats(cache::CrankNicolsonCache) = cache.stats

function OrdinaryDiffEqCore.alg_cache(
    alg::CrankNicolson,
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
    diffusion = _split_operator(f)
    diffusion isa DiffusionFunction || throw(
        ArgumentError(
            "CrankNicolson() advances FlashBang's diffusion half; it was given " *
            "$(typeof(diffusion)). Put it in the diffusion slot of the splitting tuple.",
        ),
    )
    rhs = similar(u)
    workspace = Krylov.CgWorkspace(diffusion.prepared, rhs)
    T = real(eltype(u))
    return CrankNicolsonCache(
        u, uprev, rhs, diffusion, workspace, T(alg.reltol), alg.maxiter, AlgorithmStats()
    )
end

function OrdinaryDiffEqCore.initialize!(integrator, ::CrankNicolsonCache)
    integrator.kshortsize = 0
    resize!(integrator.k, integrator.kshortsize)
    return nothing
end

function OrdinaryDiffEqCore.perform_step!(
    integrator, cache::CrankNicolsonCache, repeat_step=false
)
    (; u, uprev, t, dt) = integrator
    backend = KernelAbstractions.get_backend(u)
    _timed!(cache.stats, backend) do
        converged = _crank_nicolson!(u, uprev, cache, t, dt)
        converged || (integrator.force_stepfail = true)
        return nothing
    end
    return nothing
end

function _crank_nicolson!(u, uprev, cache::CrankNicolsonCache, t, dt)
    (; rhs, diffusion, workspace) = cache
    halfdt = dt / 2

    # b = (I + (Δt/2)P)φⁿ + Δt·Iₛₜᵢₘ(tⁿ + Δt/2)/Cₘ
    copyto!(rhs, uprev)
    mul!(rhs, diffusion.prepared, uprev, halfdt, true)
    apply_stimulus!(rhs, diffusion.stim, diffusion.xs, diffusion.Cₘ, t + halfdt, dt)

    A = _CNOperator(diffusion.prepared, halfdt)
    atol = cache.reltol * LinearAlgebra.norm(rhs)
    Krylov.cg!(workspace, A, rhs, uprev; atol, rtol=zero(atol), itmax=cache.maxiter)
    copyto!(u, Krylov.solution(workspace))

    cache.stats.iterations += Krylov.iteration_count(workspace)
    return Krylov.issolved(workspace)
end
