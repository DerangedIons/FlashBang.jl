```@meta
CurrentModule = FlashBang
```

# Things That Will Bite You

- **The stimulus sign is inverted relative to a cell model.** A protocol returns `Iₐₚₚ` in PDE form, where positive depolarizes; a CytoZoo cell model's internal stimulus is a membrane current, where negative depolarizes. See [`AbstractStimulationProtocol`](@ref). Stimulation belongs to the diffusion half — the cell model's own stimulus must be off, and [`MonodomainModel`](@ref) warns when it can tell that it is not.
- **The grid is cell-centered.** `n` cells across a length `L` put the first node at `Δx/2`, not at `0`. Comparisons against a vertex-centered reference differ at `O(Δx)`.
- **Only zero-flux boundaries.** Anything else raises an `ArgumentError` naming the face.
- **One semidiscretization per concurrent solve.** The prepared operator inside the diffusion half owns scratch buffers and is stateful and single-threaded. For a threaded `EnsembleProblem`, build each trajectory's problem from its own `semidiscretize` call inside `prob_func`; the default `prob_func` hands every trajectory the same `f`.
- **Saving is opt-in and full-state.** See [Saving Output](../guides/saving.md); observe per-step quantities with a [`StateObserver`](@ref) rather than saving every step.
- **`solve` clashes with MatrixFreeOperators'.** Both packages export a `solve`; after `using FlashBang, MatrixFreeOperators`, write `FlashBang.solve`, or import MatrixFreeOperators by name.
- **Use `Float64`.** `Float32` runs, but the ArmyHeart reference measured non-finite results across 58.6% of the ToRORd parameter space in single precision.
- **On a GPU, everything that enters a kernel must be isbits** — the cell model, the stimulus, and any spatial overrides. See [Running on a GPU](../guides/gpu.md).
