# GPU parity. Gated twice: on CUDA being installed at all (so the file survives an
# environment without it) and on a functional device (so CPU-only CI skips rather than
# fails). There is no self-hosted GPU runner yet — this is a local check.

const GPU_AVAILABLE = if Base.find_package("CUDA") === nothing
    @info "CUDA is not installed; skipping GPU tests"
    false
else
    using CUDA
    CUDA.functional() ||
        @info "CUDA is installed but no functional device is present; skipping GPU tests"
    CUDA.functional()
end

if GPU_AVAILABLE
    @testset "GPU — one step matches the CPU" begin
        # Device selection happens on the *grid*, which is what carries a backend through
        # `scalar_field`, the prepared operator's scratch, and `node_coordinates`. That is
        # the supported route in v0; adapting an already-assembled `GenericSplitFunction`
        # is not, because `ODEFunction` and `GenericSplitFunction` have no `Adapt` rules
        # and would silently pass through unchanged.
        L, n = 20.0, 256
        κ, dt = 0.1, 0.01
        stim = TransmembraneStimulationProtocol(
            (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0; nonzero_intervals=((0.0, 2.0),)
        )
        # `FHNModel` is isbits, so the whole cell model rides into the kernel by
        # value. `ToRORd` holds a heap `Vector` of parameters and does not; that is the
        # known gap this test deliberately does not paper over.
        ion = FHNModel()
        @test isbitstype(typeof(ion))

        build_gpu_cable(device) = semidiscretize(
            ReactionDiffusionSplit(MonodomainModel(; κ, ion, stim)),
            FiniteDifferenceDiscretization(),
            CartesianGrid(((0.0, L),), (n,); bc=((Neumann(), Neumann()),), device),
        )

        f_cpu = build_gpu_cable(KernelAbstractions.CPU())
        f_gpu = build_gpu_cable(CUDABackend())

        u_cpu = create_initial_condition(f_cpu)
        u_gpu = create_initial_condition(f_gpu)
        @test u_gpu isa CuArray
        @test Array(u_gpu) == u_cpu

        # Each half separately first, so a mismatch says which one.
        du_cpu, du_gpu = similar(u_cpu), similar(u_gpu)
        FlashBang._reaction_function(f_cpu)(du_cpu, u_cpu, nothing, 0.0)
        FlashBang._reaction_function(f_gpu)(du_gpu, u_gpu, nothing, 0.0)
        @test Array(du_gpu) ≈ du_cpu

        dφ_cpu, dφ_gpu = similar(u_cpu, n), similar(u_gpu, n)
        FlashBang._diffusion_function(f_cpu)(dφ_cpu, view(u_cpu, 1:n), nothing, 0.0)
        FlashBang._diffusion_function(f_gpu)(dφ_gpu, view(u_gpu, 1:n), nothing, 0.0)
        @test Array(dφ_gpu) ≈ dφ_cpu

        # Then a real (short) solve through the splitting integrator.
        tend = 20.0
        solve_cable(f, u₀) = begin
            integrator = init(
                OperatorSplittingProblem(f, u₀, (0.0, tend)),
                LieTrotterGodunov((Euler(), Euler()));
                dt,
            )
            solve!(integrator)
            return Array(getvariable(integrator.u, f, :φₘ))
        end

        φ_cpu = solve_cable(f_cpu, u_cpu)
        φ_gpu = solve_cable(f_gpu, u_gpu)
        @info "GPU parity" max_diff = maximum(abs, φ_gpu .- φ_cpu) φ_cpu_extrema = extrema(
            φ_cpu
        )
        @test φ_gpu ≈ φ_cpu rtol = 1.0e-6
        @test maximum(φ_cpu) > 0.5          # the wave actually launched
    end

    @testset "GPU — spatial overrides in the kernel" begin
        # `SpatialStep` is isbits, which is the whole reason CytoZoo has it: a closure
        # `(x, t) -> …` would not survive the trip into a kernel.
        n = 128
        overrides = (a=SpatialStep(1, 10.0, 0.05, 0.5),)
        build_override_cable(device) = semidiscretize(
            ReactionDiffusionSplit(
                MonodomainModel(; κ=0.1, ion=FHNModel()), overrides
            ),
            FiniteDifferenceDiscretization(),
            CartesianGrid(((0.0, 20.0),), (n,); bc=((Neumann(), Neumann()),), device),
        )

        f_cpu = build_override_cable(KernelAbstractions.CPU())
        f_gpu = build_override_cable(CUDABackend())
        u_cpu = create_initial_condition(f_cpu)
        setvariable!(u_cpu, f_cpu, :φₘ, 0.2)
        u_gpu = CuArray(u_cpu)

        du_cpu, du_gpu = similar(u_cpu), similar(u_gpu)
        FlashBang._reaction_function(f_cpu)(du_cpu, u_cpu, nothing, 0.0)
        FlashBang._reaction_function(f_gpu)(du_gpu, u_gpu, nothing, 0.0)

        @test Array(du_gpu) ≈ du_cpu
        # And the override is actually doing something, so parity is not parity-on-zero.
        @test any(>(0), view(du_cpu, 1:n))
        @test any(<(0), view(du_cpu, 1:n))
    end
end
