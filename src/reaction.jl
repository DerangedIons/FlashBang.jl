"""
    PointwiseODEFunction(ion, xs, overrides, nnodes, nstates, φ_symbol, state_symbol)

Reaction half of the split: the cell model `ion` evaluated independently at every node.

The solution vector is **state-blocked** (structure of arrays):

```
u = [φₘ(1:N); s₁(1:N); s₂(1:N); … ; s_{M-1}(1:N)]
```

(drawn voltage-first; the voltage is block `CytoZoo.transmembrane_potential_index(ion)`,
wherever the cell model puts it) so node `i`'s full state is the strided slice
`u[i:N:N*M]` and each *state* is contiguous. That layout is what makes the split cheap: the
diffusion half acts on `φₘ` alone, which is one contiguous block, so its dof range needs no
synchronizer object and no gather/scatter. It is also the coalesced layout on a GPU — neighbouring threads read
neighbouring nodes of the same state.

Fields beyond the kinetics — `nnodes`, `nstates`, and the two symbols — are metadata for
the solution helpers in `solution.jl`; the right-hand side itself uses only the first
three plus the two counts.

Not constructed directly; [`semidiscretize`](@ref) builds it.
"""
struct PointwiseODEFunction{I,X<:AbstractVector,O}
    ion::I
    xs::X
    overrides::O
    nnodes::Int
    nstates::Int
    φ_symbol::Symbol
    state_symbol::Symbol
end

function (f::PointwiseODEFunction)(du, u, p, t)
    _react!(du, u, f, t, KernelAbstractions.get_backend(u))
    return nothing
end

# `nothing` keeps the cell model on its non-spatial dispatch, where every spatial branch
# compiles away. Dispatching on the `Nothing` type parameter rather than testing the field
# is what makes that a compile-time choice.
@inline _node_context(::AbstractVector, ::Nothing, i) = nothing
@inline _node_context(xs::AbstractVector, overrides, i) =
    CytoZoo.SpatialContext(@inbounds(xs[i]), overrides)

# Node `i`'s state slice under the state-blocked layout: indices i, i+N, …, i+(M-1)N.
@inline _node_slice(v::AbstractVector, i::Integer, nnodes::Integer, nstates::Integer) =
    view(v, i:nnodes:(nnodes * nstates))

function _react!(du, u, f::PointwiseODEFunction, t, ::KernelAbstractions.CPU)
    N, M = f.nnodes, f.nstates
    # Nodes are independent and each thread writes a disjoint stride of `du`, so this
    # needs no synchronization. (Unlike the diffusion half's prepared operator, which is
    # stateful and must not be shared across concurrent solves.)
    Threads.@threads for i in 1:N
        f.ion(
            _node_slice(du, i, N, M),
            _node_slice(u, i, N, M),
            _node_context(f.xs, f.overrides, i),
            t,
        )
    end
    return nothing
end

# Any non-CPU backend. Launching a `Kernel{<:GPU}` and synchronizing it are *dynamic* calls
# by construction: the methods that implement them are defined by the backend package
# (CUDA.jl, AMDGPU.jl, …) and do not exist until one is loaded. Going through
# `invokelatest` says so explicitly, and keeps static analysis from reporting a missing
# method on a machine where no GPU package is present. The extra dispatch is a few tens of
# nanoseconds against a kernel launch measured in microseconds.
function _react!(du, u, f::PointwiseODEFunction, t, backend)
    kernel = _reaction_kernel!(backend)
    Base.invokelatest(
        kernel, du, u, f.ion, f.xs, f.overrides, f.nnodes, f.nstates, t; ndrange=f.nnodes
    )
    Base.invokelatest(KernelAbstractions.synchronize, backend)
    return nothing
end

@kernel function _reaction_kernel!(du, u, ion, xs, overrides, nnodes, nstates, t)
    i = @index(Global, Linear)
    ion(
        _node_slice(du, i, nnodes, nstates),
        _node_slice(u, i, nnodes, nstates),
        _node_context(xs, overrides, i),
        t,
    )
end

function Adapt.adapt_structure(to, f::PointwiseODEFunction)
    return PointwiseODEFunction(
        Adapt.adapt(to, f.ion),
        Adapt.adapt(to, f.xs),
        Adapt.adapt(to, f.overrides),
        f.nnodes,
        f.nstates,
        f.φ_symbol,
        f.state_symbol,
    )
end
