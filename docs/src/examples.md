# Examples

The examples live in `examples/` with their own environment. Run them with

```sh
julia --project=examples -t auto examples/<name>.jl
```

| Example | What it shows |
|---|---|
| `cable_1d.jl` | the [Getting Started](getting_started.md) snippet with reporting and a figure — one FitzHugh–Nagumo cable, one propagating action potential, activation times recorded by a `StateObserver` |
| `spiral_2d.jl` | a reentrant spiral from an S1–S2 cross-field protocol, Aliev–Panfilov kinetics on an 80×80 sheet |
| `niederer_benchmark.jl` | the Niederer et al. (2011) N-version benchmark — a 3D anisotropic slab with ten Tusscher–Panfilov 2006 kinetics, activation times at nine reference points against an independent finite-difference solve |
| `niederer_animation.jl` | the same slab as a video, with accumulating isochrones — nothing saved, every frame observed in flight |

The last two are ports of the corresponding MatrixFreeOperators.jl examples, which run the same physics on an adaptive block forest through a hand-rolled Godunov split. Reading a pair side by side shows what the pipeline buys and what the structured-grid restriction costs — FlashBang carries every cell at the finest spacing, where the forest carries roughly a fifth of them.

Their cell models live in `examples/` rather than in CytoZoo because they exist to drive figures. `examples/ten_tusscher_2006.jl` doubles as its own acceptance test: run it directly to check the 19-state port against published single-cell targets.
