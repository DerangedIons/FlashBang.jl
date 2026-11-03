# Adaptive refinement. Two failure modes are worth a test, and neither shows up on a uniform
# grid: a forest whose flat node order disagrees with its coordinates (every node gets the
# right physics at the wrong place, so a wave still propagates — just through a scrambled
# medium), and a regrid that loses or misplaces state, or leaves the integrator on the old
# operator. The first is caught by an unrefined forest reproducing the CartesianGrid run; the
# second by an adaptive run reproducing the uniform finest-level run's activation times.

using LinearAlgebra: norm
using MatrixFreeOperators: isselfadjoint

const AMR_EXTENT = ((0.0, 20.0), (0.0, 4.0))
const AMR_κ = 0.1
const AMR_DT = 0.05
const AMR_THRESHOLD = 0.5                     # FHN φₘ runs over roughly [0, 1]
const AMR_PROBES = ((5.0, 2.0), (10.0, 2.0), (15.0, 2.0), (19.9, 3.9))
const AMR_BC = ((Neumann(), Neumann()), (Neumann(), Neumann()))
const AMR_ALG = LieTrotterGodunov((CrankNicolson(), Euler()))
const AMR_REGRID_EVERY = 10                   # steps

function amr_split()
    stim = TransmembraneStimulationProtocol(
        (x, t) -> (t <= 2.0 && x[1] <= 1.0) ? 0.5 : 0.0; nonzero_intervals=((0.0, 2.0),)
    )
    return ReactionDiffusionSplit(MonodomainModel(; κ=AMR_κ, ion=FHNModel(), stim))
end

amr_forest(ncells, maxlevel) =
    BlockForest(CartesianGrid(AMR_EXTENT, ncells; bc=AMR_BC); blocksize=(8, 8), maxlevel)

# The node whose cell contains `x`: nearest centre, which is unambiguous for these probes.
nearest_node(f, x) = argmin(map(c -> sum(abs2, c .- x), node_coordinates(f)))

"""
Run to `tend`, regridding every `AMR_REGRID_EVERY` steps when `adaptive`, and return the
first time each probe crosses `AMR_THRESHOLD`, plus what the run did to the forest.
"""
function run_amr(forest; adaptive::Bool, tend=120.0)
    split = amr_split()
    disc = FiniteDifferenceDiscretization()
    f = semidiscretize(split, disc, forest)
    integrator = init(OperatorSplittingProblem(f, create_initial_condition(f), (0.0, tend)),
        AMR_ALG; dt=AMR_DT)
    activation = fill(NaN, length(AMR_PROBES))
    (steps, regrids, nonsymmetric, cells) = (0, 0, false, 0)
    while integrator.t < tend - AMR_DT / 2
        step!(integrator, AMR_DT, true)
        steps += 1
        cells += num_nodes(f)
        φ = getvariable(integrator.u, f, :φₘ)
        for (k, x) in enumerate(AMR_PROBES)
            isnan(activation[k]) && φ[nearest_node(f, x)] > AMR_THRESHOLD &&
                (activation[k] = integrator.t)
        end
        (adaptive && steps % AMR_REGRID_EVERY == 0) || continue
        f_old = f
        (f, u) = regrid(split, disc, f, integrator.u;
            refine=b -> maximum(norm, interior(b)) > 0.05,
            coarsen=b -> maximum(norm, interior(b)) < 0.01)
        f === f_old && continue
        regrids += 1
        nonsymmetric |= !isselfadjoint(FlashBang._diffusion_function(f).op)
        integrator = init(OperatorSplittingProblem(f, u, (integrator.t, tend)), AMR_ALG;
            dt=AMR_DT)
    end
    return (; activation, regrids, nonsymmetric, mean_cells=cells / steps)
end

@testset "Adaptive refinement" begin
    @testset "an unrefined forest reproduces the CartesianGrid run" begin
        split = amr_split()
        disc = FiniteDifferenceDiscretization()
        grid = CartesianGrid(AMR_EXTENT, (80, 16); bc=AMR_BC)
        fg = semidiscretize(split, disc, grid)
        ff = semidiscretize(split, disc, amr_forest((80, 16), 0))
        # The forest orders nodes block by block; map each forest node to the grid node at
        # the same coordinates, so the comparison is position by position.
        position = Dict(Tuple(x) => i for (i, x) in enumerate(node_coordinates(fg)))
        perm = [position[Tuple(x)] for x in node_coordinates(ff)]
        (ug, uf) = map((fg, ff)) do f
            integrator = init(
                OperatorSplittingProblem(f, create_initial_condition(f), (0.0, 30.0)),
                AMR_ALG; dt=AMR_DT,
            )
            solve!(integrator)
            return getvariable(integrator.u, f, :φₘ)
        end
        Δ = maximum(abs, ug[perm] .- uf)
        @info "forest vs CartesianGrid after 30 ms" max_abs_difference = Δ φ_range = extrema(ug)
        @test Δ ≤ 1e-12
        @test maximum(ug) > AMR_THRESHOLD        # the wave did start, so Δ compared something
    end

    @testset "an adaptive run reproduces the uniform finest-level activation times" begin
        uniform = run_amr(amr_forest((80, 16), 0); adaptive=false)
        forest = amr_forest((40, 8), 1)
        refine!(forest, x -> x[1] < 3.0)          # |∇φ| is zero at t = 0; seed the stimulus
        adaptive = run_amr(forest; adaptive=true)
        Δ = maximum(abs, adaptive.activation .- uniform.activation)
        @info "adaptive vs uniform finest level" uniform.activation adaptive.activation Δ adaptive.regrids adaptive.nonsymmetric adaptive.mean_cells uniform.mean_cells
        @test all(isfinite, uniform.activation)
        # A regrid that drops or misplaces state shifts the front by whole cells (0.25 mm, ≈
        # 1 ms at this speed); interpolation alone stays well inside one step's worth.
        @test Δ ≤ 0.2
        @test adaptive.regrids > 0
        @test adaptive.nonsymmetric               # mixed levels occurred, so GMRES ran
        @test adaptive.mean_cells < 0.8 * uniform.mean_cells
    end
end
