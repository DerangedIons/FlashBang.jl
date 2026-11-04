```@meta
CurrentModule = FlashBang
```

# Running on a GPU

## The mental model

The device is a property of the **grid**, and everything downstream inherits it. `semidiscretize` reads the backend off the grid when it allocates the node coordinates and the prepared operator's scratch fields; [`create_initial_condition`](@ref) reads it off those coordinates when it allocates `u`. So moving a model to a GPU is one keyword at the top of the pipeline, not a conversion step at the end.

Once there, the two halves of the split map onto the hardware differently:

- **Diffusion** is a stencil sweep over the voltage block — MatrixFreeOperators' prepared operator, applied on the device. It does a few flops per value it reads, so it is memory-bound, like an explicit heat-equation update.
- **Reaction** is one KernelAbstractions kernel with one thread per node. Each thread runs the cell model on its own node's state and touches no other node, so the reaction half is embarrassingly parallel — the same structure as a pointwise source term, with tens to hundreds of flops (and many `exp`s) per node. This is where most of the GPU's advantage comes from.

The [state-blocked layout](solution_vector.md) is what makes the reaction kernel fast: thread `i` reads `u[i + kN]` for each state `k`, so at every state neighbouring threads read neighbouring addresses, and the loads coalesce. An interleaved layout (node by node) would scatter each warp's loads over `M` times as much memory.

## The switch

```julia
using FlashBang, CUDA
using CytoZoo: FHNModel
using OrdinaryDiffEqLowOrderRK: Euler

grid = CartesianGrid(((0.0, 20.0),), (400,);
                     bc = ((Neumann(), Neumann()),), device = CUDABackend())

stim = TransmembraneStimulationProtocol(
    (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0;
    nonzero_intervals = ((0.0, 2.0),),
)
model = MonodomainModel(; κ = 0.1, ion = FHNModel(), stim)
f = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)

u₀ = create_initial_condition(f)       # a CuArray
prob = OperatorSplittingProblem(f, u₀, (0.0, 400.0))
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, saveat = 1.0)

φ = Array(sol[:φₘ, end])               # copy back to the host to plot or save
```

Everything except the `device` keyword and the final `Array` is the CPU pipeline unchanged. FlashBang does not depend on CUDA.jl: loading it is what supplies `CUDABackend` and the kernel-launch methods. The test suite's GPU parity checks run whenever a functional CUDA device is present and are skipped otherwise.

## What has to be isbits

A GPU kernel receives its arguments **by value**, copied into the device's parameter space. A plain struct of numbers survives that copy. Anything holding a pointer to host memory — a `Vector`, a `Dict`, a mutable struct, a captured variable that Julia has boxed — does not, because the pointer means nothing on the device. Julia's name for "survives a by-value copy" is *isbits*, and three things you hand FlashBang cross into a kernel:

| Component | Where it runs on the device | Isbits when |
|---|---|---|
| the cell model (`ion`) | the reaction kernel, once per node per step | its fields are numbers and isbits structs — `CytoZoo.FHNModel` is; `ToRORd`, whose parameters live in a heap `Vector`, is not yet |
| the stimulus `f(x, t)` | a broadcast over the node coordinates in the diffusion half | `f` is a plain function or a closure over immutable values — see below |
| spatial `overrides` | the reaction kernel, through a `CytoZoo.SpatialContext` | every value is a `CytoZoo.SpatialFunction` (`Constant`, `SpatialStep`, `SpatialGradient`); plain numbers and `(x, t) -> v` callables are CPU-only |

Check before you launch, rather than decoding a kernel compilation error afterward:

```julia
isbitstype(typeof(model.ion))      # the cell model
isbitstype(typeof(stim))           # the stimulus protocol
```

The stimulus deserves a closer look, because whether a function survives the trip depends on where its numbers come from, not on how it reads. A closure is a struct whose fields are the local variables it captures, so it is isbits when they are:

```julia
pulse(x₀, amp) = (x, t) -> (t <= 2.0 && x[1] <= x₀) ? amp : 0.0
stim = TransmembraneStimulationProtocol(pulse(1.0, 0.5); nonzero_intervals = ((0.0, 2.0),))
isbitstype(typeof(stim))   # true: two Float64s captured by value
```

Two ways to get this wrong, and the `isbitstype` check catches only the first:

- **A captured variable that is reassigned** anywhere in the enclosing function gets boxed — a heap cell shared between the closure and its scope — and the closure is no longer isbits.
- **A non-`const` global** read inside `f`, as in an anonymous function typed at the REPL that refers to `amp` defined there. Globals are not captured at all, so the closure has no fields and `isbitstype` says `true`, but the read is a lookup in host memory and the kernel fails to compile. Write literals, declare the global `const`, or pass the values in through a function like `pulse` above.

Spatial heterogeneity follows the same rule. This works on the device:

```julia
using CytoZoo: SpatialStep
split = ReactionDiffusionSplit(model, (a = SpatialStep(1, 10.0, 0.05, 0.5),))
```

whereas `(a = (x, t) -> x[1] < 10 ? 0.05 : 0.5,)` is the same physics in a form that only the CPU path accepts.

## Build on the device; do not adapt afterward

Choose the device when you build the grid. Building on the CPU and calling `Adapt.adapt(CuArray, f)` on the result does **not** work: `GenericSplitFunction` and `ODEFunction` have no `Adapt` rules, so the call returns `f` unchanged, still pointing at host arrays — and nothing errors until much later, if at all. If `create_initial_condition(f)` returns an `Array` rather than a `CuArray`, the device never reached the grid.

## Initial conditions

[`create_initial_condition`](@ref) allocates on the device and fills it with the cell model's default state. [`setvariable!`](@ref) then works in both forms:

```julia
setvariable!(u₀, f, :φₘ, 0.2)          # a constant: a device `fill!`
setvariable!(u₀, f, :φₘ) do x          # a profile: broadcast over the device coordinates
    x[1] < 1.0 ? 1.0 : 0.0
end
```

The function form runs on the device, so the isbits rule above applies to the function too. To impose a field you already have on the host — loaded from a file, or computed by a CPU-only routine — copy it into the block directly:

```julia
copyto!(view(u₀, variable_range(f, :φₘ)), φ_host)
```

The layout is identical on every device, so a whole state vector built on the CPU also transfers as-is: `CuArray(u_cpu)`.

## Reading results back

Every saved state in `sol.u` is a device array, and so is everything indexed out of the solution: `sol[:φₘ, end]` is a `CuArray`. Wrap it in `Array(…)` before plotting, writing to disk, or indexing element by element.

Saves live in **device memory**, which is usually the scarcer resource. The [costs in Saving Output](saving.md) apply unchanged — 64 MB per save for the 0.1 mm Niederer slab — but against a GPU's memory rather than the host's. A `saveat` that is harmless on a workstation can exhaust a card; save sparsely and observe the rest.

## Observers on the device

Inside a [`StateObserver`](@ref), `u` is the integrator's device buffer. The CPU activation-time loop in [Saving Output](saving.md) indexes it element by element, which on a GPU means one device-to-host transfer per element: CUDA.jl refuses it with a "scalar indexing is disallowed" error, and even where allowed it would be slower than the solve itself. Write the update as a broadcast instead, with the accumulator on the device:

```julia
activation = CUDA.fill(NaN, num_nodes(f))
record = StateObserver() do u, t
    φ = getvariable(u, f, :φₘ)
    activation .= ifelse.(isnan.(activation) .& (φ .> 0.5), t, activation)
end
sol = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt = 0.005, callback = record)
activation_host = Array(activation)
```

The same logic as the loop, now one fused kernel per step, and nothing crosses to the host until the solve is over.

When you do need data on the host during the run — movie frames, a trace for live plotting — copy a whole block into a preallocated host buffer and thin the firing with `every`:

```julia
frame = Vector{Float64}(undef, num_nodes(f))
movie = StateObserver(; every = 200) do u, t
    copyto!(frame, getvariable(u, f, :φₘ))   # one bulk transfer, then work on the host
    # ... render `frame`
end
```

Each transfer synchronizes the device with the host, so its cost is mostly the stall rather than the bytes; bulk copies every few hundred steps are cheap, per-step copies are not.

## Precision and hardware

Use `Float64`, as on the CPU: in single precision the ArmyHeart reference measured non-finite results across 58.6% of the ToRORd parameter space. That recommendation has a hardware cost worth knowing. Data-center GPUs (A100, H100) run `Float64` at half their `Float32` rate; consumer and most workstation cards run it at 1/32 to 1/64. On those cards the reaction kernel's arithmetic, not memory, becomes the bottleneck, and much of the GPU's advantage disappears. Apple's Metal does not support `Float64` at all.

The device code is written against KernelAbstractions rather than CUDA, so other backends (AMDGPU's `ROCBackend`, for instance) are not ruled out by design. CUDA is the only one that has been run.

## When the GPU is worth it

Each split step launches several kernels — the diffusion stencil, the stimulus broadcast while it is active, the reaction kernel — and the reaction half synchronizes after its launch. That is a fixed overhead of some microseconds per step, paid regardless of grid size. A 400-node cable does too little work per step to hide it, and runs faster on a threaded CPU. The GPU wins when the per-step work is large: 2D sheets and 3D slabs, and cell models with many states. Between those, time both — it is one keyword.

## Things that still apply

- **One semidiscretization per concurrent solve.** The prepared operator's scratch is device memory, and it is just as stateful there as on the CPU. Two concurrent solves must not share one `f`, on the same GPU or otherwise.
- **The stimulus sign, the cell-centered grid, zero-flux-only boundaries** — see [Things That Will Bite You](../reference/pitfalls.md). None of them change on the device.

## When it fails

| Symptom | Cause |
|---|---|
| a kernel compilation error naming a non-isbits argument | the cell model, stimulus, or an override holds host memory — check each with `isbitstype` |
| a kernel compilation error that `isbitstype` does not explain | the stimulus or a profile function reads a non-`const` global |
| "scalar indexing is disallowed" | code — usually an observer — loops over a device array element by element; broadcast instead |
| `create_initial_condition(f)` returns an `Array` | the grid was built without `device`; adapting `f` afterward does not move it |
| out of device memory partway through | too many saves; see [Reading results back](#Reading-results-back) |
| slower than the CPU | the grid is too small to amortize per-step launch overhead, or the card's `Float64` rate is the limit |
