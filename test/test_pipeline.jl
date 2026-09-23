const PIPELINE_GRID = CartesianGrid(((0.0, 10.0),), (32,); bc=((Neumann(), Neumann()),))

"A small FHN cable semidiscretization, rebuilt per testset (prepared operators are stateful)."
function fhn_pipeline(; stim=NoStimulationProtocol(), κ=0.1, overrides=nothing)
    model = MonodomainModel(; κ, ion=FHNModel(), stim)
    split = if overrides === nothing
        ReactionDiffusionSplit(model)
    else
        ReactionDiffusionSplit(model, overrides)
    end
    return semidiscretize(split, FiniteDifferenceDiscretization(), PIPELINE_GRID)
end

@testset "Pipeline — state-blocked layout and dof ranges" begin
    f = fhn_pipeline()
    n = 32

    @test num_nodes(f) == n
    @test solution_size(f) == 2n
    @test length(create_initial_condition(f)) == 2n

    # The whole point of the SoA layout: the diffusion half's dofs are the *leading
    # contiguous* block, so its range needs no gather and no synchronizer object.
    @test f.solution_indices == (1:n, 1:(2n))
    @test f.synchronizers ==
        ntuple(_ -> OrdinaryDiffEqOperatorSplitting.NoExternalSynchronization(), 2)

    @test variable_range(f, :φₘ) == 1:n
    @test variable_range(f, :v) == 1:n          # the cell model's own name for the same state
    @test variable_range(f, :states) == (n + 1):(2n) # the whole non-voltage block
    @test_throws ArgumentError variable_range(f, :not_a_state)

    # Node i's full state is the stride i, i+n, …, which is what the reaction kernel slices.
    u = collect(1.0:(2n))
    @test getvariable(u, f, :φₘ) == 1.0:n
    @test u[[1, 1 + n]] == [1.0, 1.0 + n]
end

@testset "Pipeline — create_initial_condition / getvariable / setvariable!" begin
    f = fhn_pipeline()
    n = num_nodes(f)
    ion = FHNModel()
    u₀ = create_initial_condition(f)

    @test eltype(u₀) == Float64
    for (k, value) in enumerate(CytoZoo.default_initial_state(ion))
        @test all(==(value), u₀[((k - 1) * n + 1):(k * n)])
    end

    # `getvariable` is a view: writing through it writes the solution.
    getvariable(u₀, f, :φₘ)[3] = 0.75
    @test u₀[3] == 0.75

    xs = node_coordinates(f)
    setvariable!(u₀, f, :φₘ) do x
        x[1]
    end
    @test getvariable(u₀, f, :φₘ) ≈ [x[1] for x in xs]
    @test all(==(0.0), getvariable(u₀, f, :states)) # the other block is untouched

    # `:s` is FHN's own second state, `:states` the whole non-voltage block. They coincide
    # here only because FHN has exactly one non-voltage state; the point of the plural
    # default is that the two names stay distinct on a model that has more.
    setvariable!(u₀, f, :s, -1.5)
    @test all(==(-1.5), getvariable(u₀, f, :states))
    @test getvariable(u₀, f, :φₘ) ≈ [x[1] for x in xs]

    # A single function of position cannot fill a multi-state block.
    @test_throws ArgumentError setvariable!(identity, u₀, f, :states)
    @test_throws ArgumentError setvariable!(u₀, f, :nope, 1.0)

    @test create_initial_condition(f, Float32) isa Vector{Float32}
end

@testset "Pipeline — stimulus sign convention" begin
    # The single most confusable thing in the package: the protocol is in PDE form, where
    # positive depolarizes, while the cell model's own stimulus is negative-inward. A
    # positive protocol must push φₘ *up*, and only where it is applied.
    stim = TransmembraneStimulationProtocol(
        (x, t) -> (t <= 1.0 && x[1] <= 2.0) ? 3.0 : 0.0; nonzero_intervals=((0.0, 1.0),)
    )
    f = fhn_pipeline(; stim)
    n = num_nodes(f)
    xs = node_coordinates(f)

    u = create_initial_condition(f)         # flat rest, so diffusion contributes nothing
    du = fill(NaN, n)
    FlashBang._diffusion_function(f)(du, view(u, 1:n), nothing, 0.0)

    stimulated = [x[1] <= 2.0 for x in xs]
    @test all(du[stimulated] .≈ 3.0)                 # Cₘ = 1
    @test all(iszero, du[.!stimulated])
    @test any(stimulated) && !all(stimulated)        # the test would be vacuous otherwise

    # Outside the declared window the whole evaluation is skipped, so the result is the
    # pure diffusion of a flat field: zero.
    fill!(du, NaN)
    FlashBang._diffusion_function(f)(du, view(u, 1:n), nothing, 5.0)
    @test all(iszero, du)

    # Cₘ divides the applied current.
    model = MonodomainModel(; κ=0.1, Cₘ=2.0, ion=FHNModel(), stim)
    f2 = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), PIPELINE_GRID
    )
    du2 = fill(NaN, n)
    FlashBang._diffusion_function(f2)(
        du2, view(create_initial_condition(f2), 1:n), nothing, 0.0
    )
    @test all(du2[stimulated] .≈ 1.5)
end

@testset "Pipeline — NoStimulationProtocol is a dispatch, not a branch" begin
    f = fhn_pipeline()
    n = num_nodes(f)
    diffusion = FlashBang._diffusion_function(f)

    @test diffusion.stim isa NoStimulationProtocol
    @test !FlashBang.is_active(diffusion.stim, 0.0)

    # `apply_stimulus!` on the no-stimulus path returns `du` untouched — the evaluation is
    # removed, not multiplied by zero.
    du = rand(n)
    reference = copy(du)
    @test FlashBang.apply_stimulus!(du, diffusion.stim, node_coordinates(f), 1.0, 0.0) ===
        du
    @test du == reference

    # And the whole diffusion right-hand side is allocation-free in steady state.
    u = view(create_initial_condition(f), 1:n)
    dv = similar(collect(u))
    diffusion(dv, u, nothing, 0.0)                   # warm up
    allocated = @allocated diffusion(dv, u, nothing, 0.0)
    @info "diffusion RHS allocations (bytes)" allocated
    # `Pkg.test` forces --check-bounds=yes, which ignores the @inbounds the matrix-free
    # `mul!` relies on and costs a fixed 32 bytes. The claim is about how the right-hand
    # side runs in a real solve, so only assert it when bounds checking is at its default.
    if Base.JLOptions().check_bounds == 0
        @test allocated == 0
    else
        @test_skip allocated == 0
    end
end

@testset "Pipeline — right-hand sides are type stable" begin
    f = fhn_pipeline(;
        stim=TransmembraneStimulationProtocol((x, t) -> t <= 1.0 ? 1.0 : 0.0)
    )
    n = num_nodes(f)
    u = create_initial_condition(f)
    du = similar(u)
    diffusion = FlashBang._diffusion_function(f)
    reaction = FlashBang._reaction_function(f)

    @test @inferred(diffusion(view(du, 1:n), view(u, 1:n), nothing, 0.0)) === nothing
    @test @inferred(reaction(du, u, nothing, 0.0)) === nothing

    # Same, with spatial overrides in play — the `SpatialContext` path must stay inferable.
    fo = fhn_pipeline(; overrides=(a=SpatialStep(1, 5.0, 0.05, 0.2),))
    uo = create_initial_condition(fo)
    duo = similar(uo)
    reaction_o = FlashBang._reaction_function(fo)
    @test @inferred(reaction_o(duo, uo, nothing, 0.0)) === nothing
end

@testset "Pipeline — spatial overrides reach the cell model" begin
    # `a` is FHN's excitation threshold. Splitting the cable into a low-threshold left half
    # and a high-threshold right half makes the same φₘ fire on one side and decay on the
    # other — a sign change, which no accidental pass-through could produce.
    f = fhn_pipeline(; overrides=(a=SpatialStep(1, 5.0, 0.05, 0.5),))
    n = num_nodes(f)
    xs = node_coordinates(f)

    u = create_initial_condition(f)
    setvariable!(u, f, :φₘ, 0.2)                     # between the two thresholds
    du = fill(NaN, 2n)
    FlashBang._reaction_function(f)(du, u, nothing, 0.0)

    dφ = view(du, 1:n)
    left = [x[1] <= 5.0 for x in xs]
    @test all(dφ[left] .> 0)
    @test all(dφ[.!left] .< 0)

    # Without overrides the same state gives one sign everywhere.
    fplain = fhn_pipeline()
    uplain = create_initial_condition(fplain)
    setvariable!(uplain, fplain, :φₘ, 0.2)
    duplain = fill(NaN, 2n)
    FlashBang._reaction_function(fplain)(duplain, uplain, nothing, 0.0)
    @test all(view(duplain, 1:n) .> 0)
end

@testset "Pipeline — rejected configurations" begin
    ion = FHNModel()

    # Boundary conditions: v0 is zero-flux only, and the message must name the face.
    dirichlet = CartesianGrid(((0.0, 1.0),), (8,); bc=((Dirichlet(), Neumann()),))
    err = try
        semidiscretize(
            ReactionDiffusionSplit(MonodomainModel(; κ=1.0, ion)),
            FiniteDifferenceDiscretization(),
            dirichlet,
        )
    catch e
        e
    end
    @test err isa ArgumentError
    @test occursin("low", err.msg)

    periodic = CartesianGrid(((0.0, 1.0),), (8,); bc=((Periodic(), Periodic()),))
    @test_throws ArgumentError semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ=1.0, ion)),
        FiniteDifferenceDiscretization(),
        periodic,
    )

    # An *inhomogeneous* Neumann is still not zero flux.
    flux = CartesianGrid(((0.0, 1.0),), (8,); bc=((Neumann(1.0), Neumann()),))
    @test_throws ArgumentError semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ=1.0, ion)),
        FiniteDifferenceDiscretization(),
        flux,
    )

    # Conductivity: one entry per dimension, all positive.
    @test_throws ArgumentError semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ=(1.0, 2.0), ion)),
        FiniteDifferenceDiscretization(),
        PIPELINE_GRID,
    )
    @test_throws ArgumentError MonodomainModel(; κ=-1.0, ion)
    @test_throws ArgumentError MonodomainModel(; κ=(1.0, 0.0), ion)
    @test_throws ArgumentError MonodomainModel(; κ=1.0, χ=0.0, ion)
    @test_throws ArgumentError MonodomainModel(; κ=1.0, Cₘ=-1.0, ion)
    @test_throws ArgumentError MonodomainModel(; κ=1.0, ion, φ_symbol=:u, state_symbol=:u)

    # Discretization order.
    @test_throws ArgumentError FiniteDifferenceDiscretization(; order=4)
    @test FiniteDifferenceDiscretization().order == 2

    # Stimulus windows.
    @test_throws ArgumentError TransmembraneStimulationProtocol(
        (x, t) -> 1.0; nonzero_intervals=((1.0, 0.0),)
    )
    @test_throws ArgumentError TransmembraneStimulationProtocol(
        (x, t) -> 1.0; nonzero_intervals=()
    )
end

@testset "Pipeline — warns when the cell model still carries a stimulus" begin
    # Double stimulation with opposite sign conventions is the failure this guards against,
    # and it is silent without the warning.
    live = FHNModel(; stim=Stimulus(; amplitude=-0.5))
    @test_logs (:warn,) MonodomainModel(; κ=1.0, ion=live)

    quiet = FHNModel()      # zero-amplitude default
    @test_logs MonodomainModel(; κ=1.0, ion=quiet)
end

@testset "Pipeline — stimulus protocol activity windows" begin
    always = TransmembraneStimulationProtocol((x, t) -> 1.0)
    @test FlashBang.is_active(always, -1.0)
    @test FlashBang.is_active(always, 1.0e6)

    windowed = TransmembraneStimulationProtocol(
        (x, t) -> 1.0; nonzero_intervals=((0.0, 2.0), (500.0, 502.0))
    )
    @test FlashBang.is_active(windowed, 0.0)
    @test FlashBang.is_active(windowed, 2.0)
    @test !FlashBang.is_active(windowed, 2.5)
    @test FlashBang.is_active(windowed, 501.0)
    @test !FlashBang.is_active(windowed, 600.0)

    # Stored as a Tuple, not a Vector, so the protocol stays isbits for a GPU broadcast.
    @test windowed.nonzero_intervals isa Tuple
    @test isbitstype(typeof(windowed))
    @test !FlashBang.is_active(NoStimulationProtocol(), 0.0)
    @test NoStimulationProtocol()(nothing, 0.0) == 0
end

@testset "Pipeline — node coordinates are cell-centered and match the operator ordering" begin
    grid = CartesianGrid(
        ((0.0, 4.0), (0.0, 2.0)), (4, 2); bc=ntuple(_ -> (Neumann(), Neumann()), 2)
    )
    xs = node_coordinates(grid)

    @test length(xs) == 8
    @test xs[1] ≈ [0.5, 0.5]                 # first node at Δ/2, not at the corner
    @test xs[2] ≈ [1.5, 0.5]                 # x fastest — column-major, as `flatten` orders
    @test xs[5] ≈ [0.5, 1.5]
    @test xs[end] ≈ [3.5, 1.5]

    # The same ordering MatrixFreeOperators' own flat vectors use, which is what makes
    # `du .+= stim.(xs, t)` line up with `mul!(du, P, u)`.
    field = MatrixFreeOperators.set!(
        x -> x[1] + 10 * x[2], MatrixFreeOperators.scalar_field(grid)
    )
    @test MatrixFreeOperators.flatten(field) ≈ [x[1] + 10 * x[2] for x in xs]
end
