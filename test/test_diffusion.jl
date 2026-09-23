# Analytic oracle for the *whole* pipeline. `NullIonicModel` zeroes the kinetics, so the
# monodomain equation collapses to the heat equation, whose lowest zero-flux mode decays
# in closed form:
#
#     φ(x, 0) = ∏_d cos(πx_d/L_d)   ⟹   φ(x, t) = φ(x, 0)·exp(-λt),  λ = Σ_d κ_eff,d (π/L_d)²
#
# Everything between `semidiscretize` and `getvariable` is on the hook: the dof ranges, the
# state-blocked layout, the operator assembly, and the splitting itself (the reaction half
# is exactly the identity here, so the splitting error is zero and the whole discrepancy is
# discretization).

@testset "Diffusion — 1D analytic decay through the full pipeline" begin
    L, n = 1.0, 128
    κ = 1.0e-3                       # χ = Cₘ = 1, so κ_eff == κ
    tend, dt = 100.0, 0.01

    grid = CartesianGrid(((0.0, L),), (n,); bc=((Neumann(), Neumann()),))
    model = MonodomainModel(; κ, ion=NullIonicModel())
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    # Forward Euler on the diffusion half is only conditionally stable; if this guard trips
    # the failure below would be a blow-up, not an accuracy report.
    @test dt < spacing(grid)[1]^2 / (2 * κ)

    u₀ = create_initial_condition(f)
    setvariable!(cosine_mode(grid), u₀, f, :φₘ)

    integrator = init(
        OperatorSplittingProblem(f, u₀, (0.0, tend)),
        LieTrotterGodunov((Euler(), Euler()));
        dt,
    )
    φ = getvariable(run_to_end!(integrator), f, :φₘ)

    λ = continuum_decay_rate(κ, grid)
    xs = node_coordinates(grid)
    exact = cosine_mode(grid).(xs) .* exp(-λ * tend)
    err = maximum(abs, φ .- exact)

    # Error budget, reported so a failure says which term grew. The two have *opposite*
    # sign — the discrete eigenvalue is smaller than the continuum one (under-decay) while
    # forward Euler over-decays — so at this `n` and `dt` they largely cancel and the
    # observed error is well under either. The tolerance is set to cover either term alone,
    # so it does not depend on that cancellation surviving a parameter change.
    λ_d = discrete_decay_rate(κ, grid)
    @info "1D analytic decay" n dt tend λ λ_d err spatial_budget =
        abs(λ - λ_d) * tend * exp(-λ * tend) temporal_budget =
        λ^2 * tend * dt / 2 * exp(-λ * tend)

    @test err < 1.0e-4
    @test maximum(abs, φ) < 1.0                     # the mode decays, never grows
    @test all(isfinite, φ)
end

@testset "Diffusion — 2D analytic decay through the full pipeline" begin
    Lx, Ly = 1.0, 1.0
    nx, ny = 96, 96
    κ = 1.0e-3
    tend, dt = 50.0, 0.01

    grid = CartesianGrid(
        ((0.0, Lx), (0.0, Ly)), (nx, ny); bc=ntuple(_ -> (Neumann(), Neumann()), 2)
    )
    model = MonodomainModel(; κ, ion=NullIonicModel())
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    Δ = spacing(grid)
    @test dt < 1 / (2 * κ * sum(inv, Δ .^ 2))

    u₀ = create_initial_condition(f)
    setvariable!(cosine_mode(grid), u₀, f, :φₘ)

    integrator = init(
        OperatorSplittingProblem(f, u₀, (0.0, tend)),
        LieTrotterGodunov((Euler(), Euler()));
        dt,
    )
    φ = getvariable(run_to_end!(integrator), f, :φₘ)

    λ = continuum_decay_rate(κ, grid)
    exact = cosine_mode(grid).(node_coordinates(grid)) .* exp(-λ * tend)
    err = maximum(abs, φ .- exact)

    @info "2D analytic decay" nx ny dt tend λ λ_d = discrete_decay_rate(κ, grid) err

    @test err < 1.0e-4
    @test all(isfinite, φ)
end

@testset "Diffusion — anisotropic operator is exact on the discrete cosine mode" begin
    # Sharper than the decay tests: the lowest cosine mode is an exact eigenvector of the
    # discrete zero-flux operator, so a *single application* must reproduce -λ_d·φ to
    # round-off. Anything wrong in how the per-axis conductivities are folded into
    # `κ_min∇² + Σ(κ_d - κ_min)∂²ₓ` shows up here immediately, and unlike a structural
    # check on the operator tree this does not care how the sum is arranged.
    Lx, Ly = 2.0, 1.0
    κ = (0.133, 0.0176)              # fibres along x, as in a cardiac slab
    grid = CartesianGrid(
        ((0.0, Lx), (0.0, Ly)), (64, 48); bc=ntuple(_ -> (Neumann(), Neumann()), 2)
    )
    model = MonodomainModel(; κ, χ=140.0, Cₘ=0.01, ion=NullIonicModel())
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    κ_eff = κ ./ (140.0 * 0.01)
    xs = node_coordinates(grid)
    φ = cosine_mode(grid).(xs)
    du = similar(φ)
    FlashBang._diffusion_function(f)(du, φ, nothing, 0.0)

    expected = -discrete_decay_rate(κ_eff, grid) .* φ
    err = maximum(abs, du .- expected)
    @info "anisotropic operator application" κ_eff λ_d = discrete_decay_rate(κ_eff, grid) err
    @test err < 1.0e-11

    # The isotropic special case must collapse back to a single scaled Laplacian: carrying
    # zero correction terms would cost one halo exchange per axis per apply.
    iso = MonodomainModel(; κ=(0.5, 0.5), ion=NullIonicModel())
    @test diffusion_operator(iso, grid) isa MatrixFreeOperators.Scaled
    @test diffusion_operator(iso, grid).op isa MatrixFreeOperators.Laplacian

    # ...and the anisotropic case must not.
    aniso = diffusion_operator(model, grid)
    @test aniso isa MatrixFreeOperators.Added
    @test aniso.b.op isa MatrixFreeOperators.Derivative
    @test aniso.b.op.dim == 1        # the fibre axis carries the correction
end

@testset "Diffusion — zero-flux conserves the spatial mean" begin
    # A closed domain with no source and no reaction cannot change its total: this is the
    # boundary condition itself under test, independent of any mode shape.
    grid = CartesianGrid(((0.0, 1.0),), (64,); bc=((Neumann(), Neumann()),))
    model = MonodomainModel(; κ=0.01, ion=NullIonicModel())
    f = semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )

    u₀ = create_initial_condition(f)
    setvariable!(u₀, f, :φₘ) do x
        exp(-((x[1] - 0.3)^2) / 0.01)          # off-center bump, nothing symmetric to hide behind
    end
    mean₀ = sum(getvariable(u₀, f, :φₘ)) / num_nodes(f)

    integrator = init(
        OperatorSplittingProblem(f, u₀, (0.0, 5.0)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=0.001,
    )
    φ = getvariable(run_to_end!(integrator), f, :φₘ)
    mean₁ = sum(φ) / num_nodes(f)

    @info "zero-flux mean conservation" mean₀ mean₁ drift = abs(mean₁ - mean₀)
    @test isapprox(mean₁, mean₀; rtol=1.0e-10)
    @test maximum(φ) < maximum(getvariable(u₀, f, :φₘ))     # and the bump actually spread
end
