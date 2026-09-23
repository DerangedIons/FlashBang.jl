# FlashBang.jl — Design Specification

**Status:** draft for review · **Date:** 2026-08-21

This document specifies the *target* design of FlashBang at the level of abstractions and type hierarchy. It deliberately stops above implementation detail — keyword surfaces, tolerance defaults, migration sequencing — which get filled in once this shape is settled; the one field-level artifact it carries is §4's mocked-up definitions, indicative sketches of what each concrete type owns, explicitly non-final. The v0 design was approved 2026-08-13 and is implemented; its decisions (`handoffs/2026-08-13-2019-lightning-monodomain-v0.md`) are cited as settled, never reopened, as is the v0.x growth path named by the 2026-08-18 architecture review (`handoffs/2026-08-18-1143-julia-tuneup-architecture-review.md`, items A1–A4), which this doc still resolves. This revision reframes the whole document around **CardiacAbstractions.jl** (DESIGN.md dated 2026-08-20, implemented at v0.1.0 in the DerangedIons org): the zero-dependency base package that now owns the continuum-level EP vocabulary FlashBang previously mirrored from Thunderbolt by convention. That base's DESIGN.md is the governing sibling spec for every declaration type this doc used to own — those types are cited here, not re-specified. Studied inspirations: **CardiacAbstractions.jl's DESIGN.md** (the boundary rule this doc now lives under: anything touching a mesh, grid, dof, operator, or array layout is backend), **Thunderbolt.jl** (still the naming donor for `StateBlockedLayout`, and now the sibling backend FlashBang must stay program-portable with), **Capillarium.jl's DESIGN.md** (the in-org spec format), and **Oceananigans.jl** (the "the loop needs an owner" lesson, adopted; the framework, deliberately not).

## 1. What FlashBang is

FlashBang is the structured-grid backend of a four-package stack: CardiacAbstractions.jl supplies the continuum vocabulary (model declarations, split annotations, stimulation protocols, the cell-model contract, the pipeline verbs), MatrixFreeOperators.jl supplies spatial operators, any package conforming to the base's cell contract — CytoZoo.jl first among them — supplies pointwise kinetics, and OrdinaryDiffEqOperatorSplitting supplies time splitting. Its entire value is (1) the structured-grid restriction, which buys matrix-free operators and single-source GPU execution, and (2) — reframed by the base's arrival — being one of the backends a portable program runs on: the same declaration lines that drive Thunderbolt drive FlashBang, with only the geometry and discretization lines changing. The central claim of this doc: the package gets *thinner still* — the ~290 lines of vocabulary (`models.jl`, `stimulus.jl`) leave for the base, and what remains is precisely the backend side of the base's boundary rule: layout, kernels, glue, and the two abstractions v0.x still owes itself — a first-class owner for the state-blocked layout and an owner for the time-stepping loop.

## 2. Design principles

1. **Congruent by import.** The previous revision's founding principle — congruent by convention, never by import — is retired: CardiacAbstractions ends that experiment (its DESIGN.md, settled 2026-08-20). Congruency with Thunderbolt is now structural — both backends define methods of the *same* generic functions on the *same* declaration types — and the base's zero-dependency rule means FlashBang's featherweight stack pays nothing for it.
2. **Declare, then semidiscretize.** A model is an inert declaration of the continuous problem — now a base-owned type; geometry and numerics arrive together at the base-owned `semidiscretize` verb, to which FlashBang adds the methods for its own discretization and geometry types.
3. **Hard delegation boundaries.** Continuum vocabulary belongs to CardiacAbstractions, cell kinetics to whatever conforms to its cell contract, spatial stencils to MatrixFreeOperators, time splitting to OrdinaryDiffEqOperatorSplitting; FlashBang writes none of them, and a feature that would require it to is a feature for the neighbor package.
4. **The layout has exactly one owner.** The state-blocked SoA layout is the package's central invariant; every piece of index arithmetic lives in one type, so a future batched layout is a new type, not a hunt through five files.
5. **Physics choices are typed objects in named slots, validated at construction.** Inherited for declarations from the base (which already implements keyword-only inner-constructor validation); binding on every FlashBang-owned type — recorders, layouts, functors — via the same inner-constructor rule (F2).
6. **Device is a property of the data.** Everything is array-generic with `Adapt` rules; build on CPU, `adapt` to the device, and no FlashBang type mentions CUDA.
7. **The numbers are the contract.** Analytic decay oracles, the frozen cable conduction-velocity golden master, 0 B/call RHS kernels, and full inference are acceptance criteria for every step toward this design — the vocabulary migration itself included: swapping who owns a type is a re-plumbing under frozen numbers, never a re-baseline.
8. **Public names never depend on upstream private shape.** Exports run the model; internals are namespaced; and any reach into another package's field layout happens in exactly one sanctioned accessor. Base types are read only through the base's published queries.

## 3. The grammar

```
MonodomainModel  →  ReactionDiffusionSplit  →  semidiscretize(split, disc, grid)  →  OperatorSplittingProblem  →  init / step!  →  foreach_step + recorders
  [CardiacAbstractions: continuous            [FlashBang: methods on the base verb —      (OrdinaryDiffEq-               (SciML)         [FlashBang: observation
   declaration + split annotation]             GenericSplitFunction + StateBlockedLayout]  OperatorSplitting)                              layer, v0.x — owns the loop]
```

The canonical programs — these are the docs quickstarts and the acceptance examples, and the declaration lines are now byte-portable to Thunderbolt. First, the 1D FHN cable:

```julia
using FlashBang, CytoZoo                              # FlashBang re-exports the base vocabulary; CytoZoo brings the cells

grid  = CartesianGrid(((0.0, 20.0),), (200,); bc = ((Neumann(), Neumann()),))
model = MonodomainModel(; κ = 1.0e-3, ion = FHNModel(),
    stim = AnalyticalTransmembraneStimulationProtocol(; f = (x, t) -> (t ≤ 2.0 && x[1] ≤ 1.0) ? 50.0 : 0.0,
                                                        nonzero_intervals = ((0.0, 2.0),)))
f  = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
u₀ = create_initial_condition(f)
integrator = init(OperatorSplittingProblem(f, u₀, (0.0, 100.0)),
                  LieTrotterGodunov((Euler(), Euler())); dt = 0.01)

rec = ActivationRecorder(f; threshold = 0.5)          # v0.x: replaces three hand-rolled copies
foreach_step(integrator, 100.0) do u, t               # v0.x: owns the `t < tend - dt/2` loop
    record!(rec, u, t)
end
activation_times(rec)                                 # per-node first-crossing times; NaN = never
```

The 3D Niederer benchmark — anisotropic conductivity, a many-state ionic model, the community validation case; only the two marked lines differ from the Thunderbolt version:

```julia
grid  = CartesianGrid(((0.0, 20.0), (0.0, 7.0), (0.0, 3.0)), (100, 35, 15);      # ← backend line 1
                      bc = ntuple(_ -> (Neumann(), Neumann()), 3))
model = MonodomainModel(; κ = (0.133, 0.0176, 0.0176), χ = 140.0, Cₘ = 0.01,
                          ion = TenTusscher2006(), stim = corner_stimulus)
f     = semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)   # ← backend line 2
```

The 2D spiral — S1–S2 reentry induction is nothing but a stimulus protocol with two windows, and spatial heterogeneity now enters on the model, where the base put it:

```julia
stim  = AnalyticalTransmembraneStimulationProtocol(; f = s1s2, nonzero_intervals = ((0.0, 2.0), (335.0, 337.0)))
model = MonodomainModel(; κ = 1.0e-3, ion = AlievPanfilov(), stim,
                          overrides = (celltype = x -> x[1] < 0.5 ? 0.0 : 1.0,))
```

And the single-source GPU claim as a program — the pipeline above, moved, with nothing rewritten:

```julia
f_gpu  = adapt(CuArray, f)
u₀_gpu = create_initial_condition(f_gpu)              # allocated on the device the operator lives on
```

## 4. The type system

```
# Imported from CardiacAbstractions and re-exported — owned and specified there, cited here
AbstractEPModel → MonodomainModel              # χ, Cₘ, κ, stim, ion, overrides, cell_coordinates, symbols
ReactionDiffusionSplit                         # pure split annotation — overrides no longer live here
AbstractStimulationProtocol
└── TransmembraneStimulationProtocol           # abstract now: Thunderbolt's physical meaning won the name
    ├── NoStimulationProtocol                  #   fieldless — dispatch deletes the term
    └── AnalyticalTransmembraneStimulationProtocol   # FlashBang's old concrete shape, renamed, keyword-only
AbstractCellModel + queries + cell_rhs!        # the contract FlashBang's kernels speak; CytoZoo conforms
semidiscretize, create_initial_condition       # base-owned verbs; FlashBang adds the methods below

# Owned by FlashBang — everything that touches grid, operator, or array layout
FiniteDifferenceDiscretization                 # inert numerics descriptor; literally "backend line 2"
StateBlockedLayout                             # NEW (A1) — isbits owner of N, M, the symbols, all block/stride math
DiffusionFunction{…}                           # internal RHS half: prepared MFO operator + stimulus, on block 1:N
PointwiseODEFunction{…}                        # internal RHS half: cell contract per node; carries the layout as one field
foreach_step(g, integrator, tstop; …)          # NEW (A2) — the loop owner; a function, deliberately not a type
ActivationRecorder{T}                          # NEW (A2) — first threshold-crossing times, indexed via the layout
```

The mockups below each why-paragraph are indicative sketches, not frozen field inventories: each fixes what the type *owns*; constructors, validation, and defaults stay in the implementation.

**The imported vocabulary** — `models.jl` and `stimulus.jl` leave the repo; their types are re-exported from CardiacAbstractions, which specifies them (its §4) and already implements them. Three things change shape on the way out, all settled in the base's review and adopted here without relitigation: (i) FlashBang's concrete `TransmembraneStimulationProtocol` becomes `AnalyticalTransmembraneStimulationProtocol` with keyword-only construction, and the old name becomes the abstract statement Iₛₜᵢₘ,ᵢ = Iₛₜᵢₘ,ₑ — Thunderbolt's meaning won the clash; (ii) spatial heterogeneity (`overrides`) moves from the split annotation onto the model — exactly the seam the previous revision of this doc marked and asked for, now consummated; (iii) F2's inner-constructor validation for the declarations lands in the base, already implemented there, so that traceability item closes upstream. The model additionally gains Thunderbolt's `cell_coordinates` slot (a structured-grid backend passes physical coordinates and defaults it to `nothing`) and renames `state_symbol` to `states_symbol`. No mockups here — this doc does not re-specify types it no longer owns.

**`semidiscretize` and `create_initial_condition` become methods, not functions** — FlashBang adds methods to the base's empty generics for its own discretization and geometry types, which is what turns congruency-by-convention into congruency-by-import: two backends defining methods of the same function cannot drift on its meaning. The internals read the model exclusively through the base's trait and queries — `has_pointwise_reaction_part`, `reaction_model`, `reaction_solution_symbol`, `reaction_state_symbol`, `reaction_coordinate_system` — never `isa` or field reaches, so a foreign model type that opts into the trait splits on a FlashBang grid for free. The F1 voltage-index guard now asks the base's `transmembrane_potential_index`, which is *derived* from `state_symbols`, so name and index cannot disagree; the kernels call the base's five-argument `cell_rhs!` and inherit its documented invariants (write every slot, compute in `eltype(u)`).

```julia
CardiacAbstractions.semidiscretize(split::ReactionDiffusionSplit, disc::FiniteDifferenceDiscretization, grid::CartesianGrid)
CardiacAbstractions.create_initial_condition(f::GenericSplitFunction)   # via default_initial_state, on f's device

struct FiniteDifferenceDiscretization
    order::Int                   # == 2 in v0; exists to keep the 3-arg semidiscretize open for a 4th-order stencil
end
```

**`StateBlockedLayout`** — the v0.x centerpiece, untouched by the base extraction: the base's own DESIGN.md names it as sitting exactly on the backend side of the boundary. Today the layout `u = [φₘ(1:N); s₁(1:N); …]` exists only as index arithmetic scattered across five sites, four metadata fields smuggled onto the reaction functor, and public helpers that reach through `f.functions[2].f` — upstream's *private* field layout, so an OS field rename would be a FlashBang breaking change (Hyrum's law pointed inward). The move: an isbits value type owning N, M, the two naming symbols (read off the model via the base's symbol queries at construction), and every block/stride computation (`variable_range`, state blocks, node slices); `layout(f)` extracts it as the single sanctioned reach-through; every solution helper and both kernels dispatch on it. It gives the F1 voltage-index contract a home — the guard lives where the layout is born — and it is precisely the seam the settled batching decision needs: a batched layer becomes a second layout type honoring the same accessor contract, changing nothing else.

```julia
struct StateBlockedLayout        # isbits; construction guards transmembrane_potential_index(ion) == 1 (F1, base-derived)
    nnodes::Int                  # N
    nstates::Int                 # M
    φ_symbol::Symbol             # from reaction_solution_symbol(model)
    states_symbol::Symbol        # from reaction_state_symbol(model)
end
# owns every block/stride computation as methods (variable ranges, state blocks, node slices);
# layout(f) is the single sanctioned reach into the GenericSplitFunction internals
```

**`foreach_step` + `ActivationRecorder`** — the observation layer, kept deliberately dumb. The `while integrator.t < tend - dt/2; step!(…)` loop is currently hand-rolled in six places and first-crossing activation recording reimplemented in three; the most-duplicated code in the repo is the package's very first README snippet. `foreach_step` owns the loop and its floating-point half-step guard and yields `(u, t)` after each step; `ActivationRecorder` is a typed recorder that watches the voltage block through the layout and stores first upward threshold crossings. No abstract observer hierarchy, no scheduling framework, no SciML solution interface pretensions (the no-`sol` decision stands) — the deferred conduction-velocity and pseudo-ECG readouts become ordinary consumers of the recorded times when they arrive.

```julia
foreach_step(g, integrator, tstop)   # a function, deliberately not a type: owns the `t < tend − dt/2` loop, yields (u, t)

struct ActivationRecorder{T,A<:AbstractVector{T}}   # immutable-with-buffer shape per §9 Q3 recommendation
    layout::StateBlockedLayout
    threshold::T
    times::A                     # per-node first upward crossing; NaN = never (record! mutates only this)
end
```

**`DiffusionFunction` / `PointwiseODEFunction`** — internal RHS halves, unexported, reached via public `diffusion_function(f)` / `reaction_function(f)` accessors (promoted from the underscore names 16 test sites already use). The stimulus slot now holds a base protocol and consumes the base's `is_active` window hint; the `ion` slot holds anything satisfying the base's cell contract — the kernels speak `cell_rhs!`, not any one cell package's API. The `overrides` field is now filled from `model.overrides` (the base's slot) at semidiscretize time; how it reaches the cell model — currying via a `bind_overrides`-style hook versus a widened signature — is the base's open question Q1, and FlashBang tracks its resolution rather than deciding it here. The functors' *types* are the OS-facing adapters and stay regardless of §5's execution-engine decision.

```julia
struct DiffusionFunction{P,O,G,S<:TransmembraneStimulationProtocol,T,X<:AbstractVector}
    prepared::P                  # the MFO PreparedOperator mul! applies; stateful — one per solve
    op::O                        # unprepared operator: Adapt re-`prepare`s from it on the target device
    grid::G
    stim::S                      # base protocol; skipped via CardiacAbstractions.is_active
    Cₘ::T                        # stimulus enters as Iₛₜᵢₘ/Cₘ
    xs::X                        # node coordinates the stimulus is evaluated at
end

struct PointwiseODEFunction{I,X<:AbstractVector,O}
    ion::I                       # anything satisfying the base cell contract; kernels call cell_rhs!
    xs::X
    overrides::O                 # from model.overrides; delivery mechanism tracks base Q1
    layout::StateBlockedLayout   # replaces the four smuggled metadata fields (nnodes, nstates, 2 symbols)
end
```

## 5. Alternatives considered

**Keeping the vocabulary: congruency by convention vs importing CardiacAbstractions.** The previous revision's founding principle, and it genuinely earned its two years — zero coupling survived Thunderbolt's 0.0.x churn precisely because nothing imported anything. It is ended by the base's own DESIGN.md (settled 2026-08-20, with the Thunderbolt core developer's agreement to depend on the base — the fact that dissolved the original rationale), and this doc cites that decision rather than relitigating it. What FlashBang specifically gains: ~290 lines and two files deleted, drift-proof verbs, and cell-model interchangeability — CytoZoo's zoo and Thunderbolt's ionic models become interchangeable under the shared contract, so FlashBang's test suite can borrow either. What it accepts: a release coupling to the base, mitigated by the base's zero-dependency rule and its rare-coordinated-breaking policy.

**Owning the time-splitting driver vs staying on OrdinaryDiffEqOperatorSplitting (A4).** The measured 3× RHS tax (P1) plus OS's warts (no solution interface, fixed-step usage everywhere) make a self-owned stepper genuinely attractive — for what FlashBang uses, it is ~100 lines, and MFO's own examples hand-roll exactly it. It loses because the OS alg-tuple is the settled Rush–Larsen seam: RushLarsenSolvers drops into `LieTrotterGodunov((diff_alg, cell_alg))` with no type changes, and owning a stepper reopens that. Decision: P1 is treated as an upstream bug first (reproducer filed from the review's counting-functor harness); if upstream stalls, the fallback is a minimal non-FSAL forward-Euler substepper that stays *inside* the OS framework; full driver ownership is recorded as last resort so it is never rediscovered from scratch.

**Reaction-half execution: hand-rolled kernels vs delegating to LockstepODE.jl.** The reaction half — N independent per-node cell ODEs advanced in lockstep — is exactly LockstepODE's Batched mode, and the state-blocked layout is byte-identical to its `PerIndex` ordering with ode = node and ode_size = M, so the data seam already matches: no copies, no permutation. The case for delegating: principle 3 says FlashBang writes no neighbor code, yet the reaction functor's threaded CPU loop and KA kernel are precisely LockstepODE's job description — delegation turns the review's serial-fallback and GPU-launch-overhead findings (P2, P3) into already-solved problems in the package whose whole job is that loop, and buys the AMDGPU/Metal/oneAPI backends for free. The case against going all the way: the OS split consumes an RHS *function* (the alg tuple's `Euler()` owns the substepping) while LockstepODE's public surface is problem/integrator-level and owns its own timestepping, so nesting its integrator inside a substep reopens A4 and threatens the Rush–Larsen alg-tuple seam; and LockstepODE hard-depends on full OrdinaryDiffEq where FlashBang deliberately carries only the lean splitting package. Decision: the middle path — `PointwiseODEFunction` stays as the thin OS-facing adapter the `GenericSplitFunction` contract requires, and its execution engine (the node loop and kernels) delegates to LockstepODE's batched machinery, contingent on (1) the delegated hot path holding the 0 B/call + full-inference contract under the golden masters and (2) LockstepODE exposing its RHS-level engine without the full OrdinaryDiffEq dependency riding along; until both hold, the hand-rolled kernels stay.

**A framework observation layer (Oceananigans-style `Simulation` + callbacks + schedules) vs the dumb primitive.** The framework earns its keep when run control has many axes (output writers, wizards, schedules); FlashBang has one loop shape and one recorder today, and the OS integrator does not implement the SciML solution interface a callback framework would want to stand on. A function plus one concrete recorder deletes the duplication now and leaves every richer design open; an interface gets designed when a second recorder exists (rule: no interface before a second implementer).

**Layout as functions/traits vs a value type.** Free functions are what exists today, and the review documents the cost: five scattered arithmetic sites and helpers coupled to upstream field layout. A trait cannot carry N and M. The isbits value type costs one field on the reaction functor and buys a single owner, kernel-passability, and the batching seam.

**Returning a FlashBang-owned wrapper from `semidiscretize` vs the bare `GenericSplitFunction`.** The wrapper would make `layout(f)` a plain field read and hide upstream entirely; it loses (as a recommendation — §9 Q1) because the bare return is the congruency contract itself — the result drops into `OperatorSplittingProblem` because it *is* the OS type — and the layout accessor plus public half-accessors already reduce the upstream coupling to one sanctioned site.

**AoS layout, synchronizer objects, per-cell stimulus** — settled against in v0 (approved 2026-08-13) for SoA's contiguous diffusion view, GPU coalescing, and the uniform-stimulus convention; cited here only so this doc is self-contained.

## 6. Cross-cutting rules

**Layout contract.** Voltage is state 1: `semidiscretize` guards `transmembrane_potential_index(ion) == 1` at setup and throws otherwise (F1 — today a model with a non-leading voltage index silently diffuses a gating variable). The index is the base's symbol-derived one, so the guard cannot disagree with the model's published names; it lives with `StateBlockedLayout` construction, and supporting other indices later is a layout generalization, not an API change.

**Validation.** Declaration-level validation (positivity, symbol distinctness, interval well-formedness) lives in the base's inner constructors — implemented, closing F2 upstream. Every FlashBang-owned type follows the same rule locally: construction-time checks in inner constructors, keyword outers as thin forwards. Per the base's contract, coefficient and override values FlashBang cannot interpret are rejected at `semidiscretize`, never at solve time.

**Exports.** FlashBang re-exports the base vocabulary plus the grid and splitting vocabulary, so `using FlashBang` stays one-stop and users never import CardiacAbstractions directly; on top ride the FlashBang-owned v0.x names (`StateBlockedLayout` (Q2), `layout`, `foreach_step`, `ActivationRecorder`/`record!`/`activation_times`, `diffusion_function`, `reaction_function`, `FiniteDifferenceDiscretization`). One deliberate break rides the migration: `TransmembraneStimulationProtocol` stays exported but becomes the base's abstract type, and v0.x call sites move to `AnalyticalTransmembraneStimulationProtocol(; f, …)` — acceptable while unregistered, done once, with the vocabulary swap. `solve` stays deliberately unexported. Internals stay namespaced.

**Cell models live in CytoZoo (A3), reached through the base contract.** `AlievPanfilov` and `TenTusscher2006` graduate from `examples/` to CytoZoo via PRs (TT06's self-acceptance test becomes a real CytoZoo test); CytoZoo re-roots under the base's `AbstractCellModel`, making its whole zoo — and Thunderbolt's — usable in FlashBang. FlashBang's core code speaks only the base's cell contract; whether CytoZoo remains a hard dependency at all is Q5.

**Invariants.** The frozen cable CV + last-node activation time (measured from unmodified code, 2–5% band) is the coupled-path golden master; the analytic decay oracles are the diffusion golden master; both RHS halves stay 0 B/call (beyond the documented thread-task overhead) and fully inferred; species of change that would move any golden number are rejected, never re-baselined — the vocabulary migration is explicitly gated on this. The DiffEq `p` argument is deliberately ignored by both halves — parameters live in functor fields — and is documented as such.

## 7. Non-goals

**Multi-physics design (bidomain, mechanics).** The seam now lives in the base — `AbstractEPModel` and the trait grammar are CardiacAbstractions' to extend, and its own doc defers bidomain until a backend can discretize it; FlashBang designs nothing here. **Batched multi-simulation runs** — composes later as a second layout type over the A1 seam; nothing to design until then. **CV / activation-map / pseudo-ECG readouts** — future consumers of the recorder, deliberately not designed in §4. **Rush–Larsen** — slots into the OS alg tuple with no type changes; the shared gate contract is deferred by the base for the same reason. **A SciML solution object** — settled no; the integrator is driven, not collected. **AMR, multi-GPU, registration, migration sequencing** — out of scope here; sequencing lives in the review's ordered fix list. (Spatially varying κ, formerly in this list, is no longer a free choice — the base's portable-field contract reaches it; see Q7.)

## 8. Ecosystem alignment

FlashBang becomes the first consummated backend of the CardiacAbstractions school — the SciMLBase/CommonSolve pattern of a zero-dep base owning declarations and verb contracts, heavyweight implementers adding methods. Alignment with Thunderbolt stops being a discipline FlashBang maintains by hand and becomes a property of the import: the acceptance test is the base doc's own — a user program that switches between the two backends by changing only its geometry and discretization lines, which is why §3's canonical programs mark those lines explicitly. Everything this doc owns (`StateBlockedLayout`, the functors, the observation layer, the MFO glue) sits on the backend side of the base's boundary rule, confirmed by the base's DESIGN.md naming them as untouched. Where FlashBang is deliberately narrower than the base's portable-field contract for now is surfaced honestly in Q7 rather than papered over.

## 9. Open questions

1. **`semidiscretize` return type.** (a) Keep the bare `GenericSplitFunction` + public `layout`/half-accessors; (b) a thin FlashBang wrapper forwarding the OS interface. **Recommend (a)** — the bare return is the congruency contract, and A1 already collapses the coupling to one accessor (§5 gives the full trade).
2. **`StateBlockedLayout` export status.** (a) Exported, like Thunderbolt's; (b) namespaced, reached only through `layout(f)`. **Recommend (a)**: the batching layer and any external recorder will dispatch on it, and the printed type is useful configuration reporting.
3. **Recorder mutability shape.** (a) `ActivationRecorder` is a mutable struct mutated by `record!`; (b) immutable struct holding a times buffer. **Recommend (b)** — immutable-with-buffer matches the package's style (functors with array fields), adapts cleanly, and `record!` mutates only the buffer.
4. **`foreach_step` sampling.** (a) Yield after every step; (b) accept a `stride`/`ts` so movie-writing examples don't record every dt. **Recommend (a) now** — the recorder decides what to keep, and a stride keyword is an additive change later if profiling demands it.
5. **CytoZoo dependency status.** With kernels speaking the base's cell contract, FlashBang's core no longer calls CytoZoo. (a) Drop CytoZoo to a test/examples dependency — users bring their own cell package, as the base's canonical `using FlashBang, CytoZoo` already implies; (b) keep the hard dependency and re-export a starter zoo for out-of-the-box quickstarts. **Recommend (a)** — it makes the delegation boundary real, and the quickstart cost is one `using`.
6. **Where `Adapt` rules for base-owned types live.** FlashBang today defines `adapt_structure` for its own concrete protocol; post-migration that type is the base's, and defining the rule in FlashBang is type piracy. (a) The base gains an `Adapt` package extension (a weak dep keeps its zero hard-dependency rule intact); (b) FlashBang adapts field-wise inside its own functors' `Adapt` rules, never touching the foreign type. **Recommend (a)** — one rule serving every backend beats per-backend reconstruction, but it is a base-repo decision needing both maintainers per its dependency rule.
7. **The portable-field contract vs FlashBang's coefficient support.** The base's §6 declares `Number` *and callables* portable in every coefficient slot, "honored by every backend" — but FlashBang today supports neither callable κ nor callable χ/Cₘ, and the previous revision listed spatially varying κ as a non-goal. (a) Honor callable κ in v0.x via MFO's existing `divergence∘scaling∘gradient` composition, reject callable χ/Cₘ at `semidiscretize` with a clear error, and ask the base to scope its hard guarantee to κ and overrides until a backend needs more; (b) implement the full contract now. **Recommend (a)** — it meets the contract where the machinery already exists and turns the remainder into an honest, visible gap negotiated in the base rather than a silent one.

## 10. Traceability

| Spec element | Resolves |
|---|---|
| Vocabulary types extracted to CardiacAbstractions, re-exported (§4) | Protocol name clash with Thunderbolt; overrides-on-split seam; congruency drift the old principle 1 could not prevent |
| `StateBlockedLayout` + `layout(f)` single reach-through | A1; the Hyrum reach-through in solution.jl; smuggled metadata fields |
| Voltage-index guard at layout construction, base-derived | F1 |
| Inner-constructor validation rule | F2 — closed upstream for declarations (implemented in the base); binding locally for FlashBang-owned types |
| `foreach_step` + `ActivationRecorder` | A2; loop ×6 and activation-recording ×3 duplication |
| Cell models graduate to CytoZoo through the base contract | A3; partially T3 (many-state coverage); base doc's interchangeability claim |
| Splitting stance: upstream-first, in-framework fallback, ownership last resort | A4, P1 |
| Public `diffusion_function`/`reaction_function` | 16 test-site reaches into underscore internals |
| Golden-master invariants (§6), migration gated on them | T5; review gotcha "the numbers never move" |
| Reaction-engine stance: adapter kept, LockstepODE delegation contingent (§5) | P2, P3 (outsourced when the delegation lands) |
