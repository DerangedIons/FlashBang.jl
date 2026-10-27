# The two structure-aware inner algorithms and the voltage-index layout fix. Each testset is
# built so the property under test has a closed form or a sharp contrast with `Euler()`,
# rather than a tolerance that a subtly wrong implementation could still meet.

"Zero-flux 1D cable of `n` cells on `[0, L]`."
function algorithm_cable(n; L=1.0)
    return CartesianGrid(((0.0, L),), (n,); bc=((Neumann(), Neumann()),))
end

@testset "RushLarsen — the cell model's own step is what runs" begin
    # The gate obeys dw/dt = -r·w and its Rush–Larsen step is the exact exponential, so a
    # solver that really calls the hook reproduces w₀(x)·exp(-r·t) to rounding at *any* Δt.
    # Euler at the same coarse Δt compounds (1 - rΔt) per step instead and misses by tens
    # of percent. The gate varies in x, so a node-slicing error cannot hide behind a
    # uniform field.
    r, tend, n = 1.0, 5.0, 64
    grid = algorithm_cable(n)
    f = semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ=1.0e-3, ion=GatedNullModel(r))),
        FiniteDifferenceDiscretization(),
        grid,
    )
    w₀(x) = 1 + x[1]
    exact = w₀.(node_coordinates(f)) .* exp(-r * tend)

    function gate_at_end(reaction_alg, dt)
        local u₀ = create_initial_condition(f)
        setvariable!(w₀, u₀, f, :w)
        # `local`: these helpers are closures, and a bare assignment would rebind the
        # testset's own `integrator`.
        local integrator = init(
            OperatorSplittingProblem(f, u₀, (0.0, tend)),
            LieTrotterGodunov((Euler(), reaction_alg));
            dt,
        )
        solve!(integrator)
        return (copy(getvariable(integrator.u, f, :w)), integrator)
    end

    for dt in (0.5, 0.05)
        (w, integrator) = gate_at_end(RushLarsen(), dt)
        rel_err = maximum(abs, (w .- exact) ./ exact)
        (_, ode) = algorithm_stats(integrator)
        @info "RushLarsen vs exact exponential" dt rel_err ode.steps ode.elapsed_ns
        @test rel_err < 1.0e-13
        @test ode.steps == round(Int, tend / dt)
    end

    (w_euler, _) = gate_at_end(Euler(), 0.5)
    euler_err = maximum(abs, (w_euler .- exact) ./ exact)
    @info "Euler at the same coarse step" dt = 0.5 rel_err = euler_err
    @test euler_err > 0.5

    # A cell model without a Rush–Larsen step is refused at `init`, not mid-solve.
    f_fhn = semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ=1.0e-3, ion=FHNModel())),
        FiniteDifferenceDiscretization(),
        grid,
    )
    @test_throws ArgumentError init(
        OperatorSplittingProblem(f_fhn, create_initial_condition(f_fhn), (0.0, 1.0)),
        LieTrotterGodunov((Euler(), RushLarsen()));
        dt=0.1,
    )
end

@testset "CrankNicolson — matches Euler where both work, stable where Euler is not" begin
    κ, n = 1.0e-3, 64
    grid = algorithm_cable(n)
    h = spacing(grid)[1]
    dt_explicit = h^2 / (2κ)
    f_for() = semidiscretize(
        ReactionDiffusionSplit(MonodomainModel(; κ, ion=NullIonicModel())),
        FiniteDifferenceDiscretization(),
        grid,
    )

    function diffuse(diffusion_alg, φ₀, dt, nsteps)
        local f = f_for()
        local u₀ = create_initial_condition(f)
        setvariable!(φ₀, u₀, f, :φₘ)
        local integrator = init(
            OperatorSplittingProblem(f, u₀, (0.0, nsteps * dt)),
            LieTrotterGodunov((diffusion_alg, Euler()));
            dt,
        )
        solve!(integrator)
        return (copy(getvariable(integrator.u, f, :φₘ)), integrator)
    end

    # Accuracy: the lowest cosine mode is an exact eigenvector of the discrete operator, so
    # both schemes are measured against exp(-λ_d t) with no spatial error in the way.
    # Crank–Nicolson is second order, Euler first, so CN must be much the closer of the two.
    dt, nsteps = dt_explicit / 2, 400
    mode = cosine_mode(grid)
    exact =
        mode.(node_coordinates(grid)) .* exp(-discrete_decay_rate(κ, grid) * nsteps * dt)
    (φ_cn, integrator) = diffuse(CrankNicolson(), mode, dt, nsteps)
    (φ_eu, _) = diffuse(Euler(), mode, dt, nsteps)
    (pde, _) = algorithm_stats(integrator)
    err_cn = maximum(abs, φ_cn .- exact)
    err_eu = maximum(abs, φ_eu .- exact)
    cn_vs_euler = maximum(abs, φ_cn .- φ_eu)
    mean_cg = pde.iterations / pde.steps
    @info "CN vs Euler below the explicit limit" dt dt_explicit err_cn err_eu cn_vs_euler mean_cg
    @test err_eu < 1.0e-4
    @test err_cn < err_eu / 100
    @test pde.steps == nsteps

    # Stability: a step profile excites every grid mode, and at 10× the explicit limit
    # Euler amplifies the highest one by |1 - 4κΔt/h²| = 19 per step. Crank–Nicolson's
    # amplification factor is below 1 in magnitude for every mode, so the L² norm cannot
    # grow, and zero flux conserves the mean.
    dt, nsteps = 10 * dt_explicit, 40
    step_profile(x) = x[1] < 0.5 ? 1.0 : 0.0
    φ₀ = step_profile.(node_coordinates(grid))
    (φ_cn, integrator) = diffuse(CrankNicolson(), step_profile, dt, nsteps)
    (φ_eu, _) = diffuse(Euler(), step_profile, dt, nsteps)
    (pde, _) = algorithm_stats(integrator)
    cn_extrema = extrema(φ_cn)
    euler_max_abs = maximum(abs, φ_eu)
    mean_cg = pde.iterations / pde.steps
    ms_per_step = pde.elapsed_ns / pde.steps / 1.0e6
    @info "CN vs Euler at 10× the explicit limit" dt dt_explicit cn_extrema euler_max_abs pde.iterations mean_cg ms_per_step
    @test all(isfinite, φ_cn)
    @test sqrt(sum(abs2, φ_cn)) <= sqrt(sum(abs2, φ₀))
    @test isapprox(sum(φ_cn), sum(φ₀); rtol=1.0e-8)
    @test !all(isfinite, φ_eu) || euler_max_abs > 1.0e3
    # Unpreconditioned CG on I - (Δt/2)P: condition number 1 + 2κΔt/h² = 11 here, so a few
    # tens of iterations reach 1e-10, nowhere near the cap.
    @test 0 < mean_cg < 100
end

@testset "Layout — a voltage that is not state 1 is the block that diffuses" begin
    # gotranx orders states by dependency, so the voltage is rarely first. With the voltage
    # second, block 1 is a gate: frozen here (rate 0) at a non-uniform profile that diffusion
    # would visibly smear. Before the fix the diffusion half took `1:N` regardless, so it
    # diffused the gate and left the voltage untouched.
    n, κ, tend, dt = 128, 1.0e-3, 100.0, 0.01
    grid = algorithm_cable(n)
    f = semidiscretize(
        ReactionDiffusionSplit(
            MonodomainModel(; κ, ion=GatedNullModel(0.0; voltage_index=2))
        ),
        FiniteDifferenceDiscretization(),
        grid,
    )

    @test variable_range(f, :φₘ) == (n + 1):(2n)
    @test variable_range(f, :w) == 1:n
    @test f.solution_indices[1] == (n + 1):(2n)
    @test_throws ArgumentError variable_range(f, :states)

    u₀ = create_initial_condition(f)
    setvariable!(cosine_mode(grid), u₀, f, :φₘ)
    gate(x) = x[1] < 0.5 ? 1.0 : 0.0
    setvariable!(gate, u₀, f, :w)

    integrator = init(
        OperatorSplittingProblem(f, u₀, (0.0, tend)),
        LieTrotterGodunov((Euler(), Euler()));
        dt,
    )
    u = run_to_end!(integrator)
    φ = getvariable(u, f, :φₘ)
    w = getvariable(u, f, :w)

    exact =
        cosine_mode(grid).(node_coordinates(grid)) .*
        exp(-continuum_decay_rate(κ, grid) * tend)
    err = maximum(abs, φ .- exact)
    gate_drift = maximum(abs, w .- gate.(node_coordinates(grid)))
    @info "voltage at index 2" err gate_drift
    @test err < 1.0e-4
    @test w == gate.(node_coordinates(grid))
end
