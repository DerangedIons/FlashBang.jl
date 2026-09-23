# Isochrone animation of the Niederer et al. (2011) N-version benchmark.
# Phil. Trans. R. Soc. A 369:4331-4351.
#
# Same physics, discretization, and pipeline as `examples/niederer_benchmark.jl` — see the
# header there for the full story. This script trades the static snapshot panels for a
# video: the z = 1.5 mm mid-plane transmembrane potential sweeping the slab, overlaid with
# *accumulating isochrones* — contour lines of the running first-activation-time field at
# 5 ms intervals. Cells that have not yet crossed 0 mV hold NaN in the activation buffer,
# so contours only ever appear in tissue the wavefront has already visited; the wake of
# the wave fills in with its own history as it propagates.
#
# The run stops at t = 50 ms: full-resolution activation completes by ~45 ms, and the
# remaining plateau adds frames without adding information.
#
# Run with:      julia --project=examples -t auto examples/niederer_animation.jl
# Fast check:    NIEDERER_SMOKE=true  ...   (0.4 mm, 12 ms; seconds, exercises every path)
# Custom:        NIEDERER_DX=0.2 NIEDERER_TEND=50 NIEDERER_FRAME_EVERY=40 ...

using Pkg
Pkg.activate(@__DIR__)

using FlashBang
using OrdinaryDiffEqLowOrderRK: Euler
using CairoMakie, Printf

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

#--------------------------------------------------------------------------------# Physiology (Niederer 2011 Table 3)

const LX, LY, LZ = 20.0, 7.0, 3.0    # slab dimensions (mm); fibres along x

const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)   # monodomain harmonic mean, 0.133418 mS/mm
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)   #                           0.017606 mS/mm

const BETA = 140.0                   # surface-to-volume ratio (1/mm)
const CM = 0.01                      # membrane capacitance (µF/mm²)

const D_L = σ_L / (BETA * CM)
const D_T = σ_T / (BETA * CM)

const STIM_I = 50.0 / BETA           # µA/mm², positive = depolarizing
const STIM_DUR = 2.0                 # ms
const STIM_SIZE = 1.5                # mm
const V_ACTIVATION = 0.0             # mV

#--------------------------------------------------------------------------------# Discretization

const SMOKE = get(ENV, "NIEDERER_SMOKE", "false") == "true"

const DX = parse(Float64, get(ENV, "NIEDERER_DX", SMOKE ? "0.4" : "0.1"))
# 50 ms, not the benchmark's 60: activation is complete by ~45 ms and the animation has
# nothing left to show after that.
const TEND = parse(Float64, get(ENV, "NIEDERER_TEND", SMOKE ? "12.0" : "50.0"))
const DT = 0.01                      # ms
const SAMPLE_EVERY = 10              # steps ≡ 0.1 ms activation-time quantization

# One video frame per FRAME_EVERY solver steps. The full run takes 5 000 steps, so 40
# gives 125 frames — ~6 s at 20 fps. Kept a multiple of SAMPLE_EVERY so the isochrone
# buffer is freshly updated on every frame.
const FRAME_EVERY = parse(Int, get(ENV, "NIEDERER_FRAME_EVERY", SMOKE ? "20" : "40"))

const NCELLS = (round(Int, LX / DX), round(Int, LY / DX), round(Int, LZ / DX))
const Z_SLICE = 1.5                  # mid-plane (mm)
const ISO_LEVELS = collect(5.0:5.0:45.0)   # isochrone levels (ms)

outpath(name) = joinpath(@__DIR__, SMOKE ? "niederer_smoke_$name" : "niederer_$name")

#--------------------------------------------------------------------------------# Problem

const STIMULUS = TransmembraneStimulationProtocol(
    (x, t) -> (x[1] <= STIM_SIZE && x[2] <= STIM_SIZE && x[3] <= STIM_SIZE) ? STIM_I : 0.0;
    nonzero_intervals=((0.0, STIM_DUR),),
)

function build_slab()
    grid = CartesianGrid(
        ((0.0, LX), (0.0, LY), (0.0, LZ)), NCELLS; bc=ntuple(_ -> (Neumann(), Neumann()), 3)
    )
    model = MonodomainModel(;
        κ=(σ_L, σ_T, σ_T), χ=BETA, Cₘ=CM, ion=TenTusscherEpi(), stim=STIMULUS
    )
    return grid,
    semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
end

#--------------------------------------------------------------------------------# Solve

"""
Step the split integrator to `TEND`, maintaining a full-field first-activation-time buffer
(NaN until a cell first crosses `V_ACTIVATION`), and call
`on_frame(V::Array{3}, act::Array{3}, t)` every `frame_every` steps with the potential and
the activation buffer reshaped to the grid. The NaNs are the masking: Makie's `contour!`
draws nothing through NaN cells, so not-yet-activated tissue stays contour-free.
"""
function simulate(grid, f; on_frame=nothing, frame_every=FRAME_EVERY)
    integrator = init(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, TEND)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=DT,
    )

    activation = fill(NaN, prod(NCELLS))
    nstep, nframes = 0, 0
    tstart = time_ns()

    while integrator.t < TEND - DT / 2
        step!(integrator, DT, true)
        nstep += 1

        if nstep % SAMPLE_EVERY == 0
            φ = getvariable(integrator.u, f, :φₘ)
            @inbounds for i in eachindex(activation)
                isnan(activation[i]) &&
                    φ[i] > V_ACTIVATION &&
                    (activation[i] = integrator.t)
            end
        end

        if on_frame !== nothing && nstep % frame_every == 0
            φ = collect(getvariable(integrator.u, f, :φₘ))
            on_frame(
                reshape(φ, NCELLS...), reshape(copy(activation), NCELLS...), integrator.t
            )
            nframes += 1
        end
    end

    wall = (time_ns() - tstart) * 1.0e-9
    @printf "  %d steps, %d frames in %.1f s (%.2f ms/step)\n" nstep nframes wall (
        1.0e3 * wall / nstep
    )
    return wall
end

#--------------------------------------------------------------------------------# Animation

function slice_axes(grid)
    Δ = spacing(grid)
    return (
        range(Δ[1] / 2, LX - Δ[1] / 2; length=NCELLS[1]),
        range(Δ[2] / 2, LY - Δ[2] / 2; length=NCELLS[2]),
    )
end

#--------------------------------------------------------------------------------

grid, f = build_slab()
xs, ys = slice_axes(grid)
kz = clamp(ceil(Int, Z_SLICE / spacing(grid)[3]), 1, NCELLS[3])

@printf "Niederer 2011 isochrone animation — %s\n" (
    SMOKE ? "SMOKE mode (coarse and short; not the benchmark answer)" : "full run"
)
@printf "  %d × %d × %d cells at Δx = %.3f mm, dt = %.3f ms to t = %.0f ms, %d threads\n" NCELLS[1] NCELLS[2] NCELLS[3] DX DT TEND Threads.nthreads()
@printf "  one frame per %d steps → %d frames at 20 fps\n\n" FRAME_EVERY (
    round(Int, TEND / DT) ÷ FRAME_EVERY
)

Δ = spacing(grid)
dt_max = 1 / (2 * (D_L / Δ[1]^2 + D_T / Δ[2]^2 + D_T / Δ[3]^2))
DT < dt_max || error("dt = $DT exceeds the explicit stability limit $dt_max for Δx = $DX")

fig = Figure(; size=(1060, 520), fontsize=17)
ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x (mm)", ylabel="y (mm)")
Colorbar(
    fig[1, 2];
    colorrange=(-90, 40),
    colormap=:inferno,
    label="transmembrane potential φₘ (mV)",
)
Label(
    fig[0, 1:2],
    "Niederer et al. 2011 N-version benchmark — FlashBang.jl";
    fontsize=23,
    font=:bold,
)
Label(
    fig[2, 1:2],
    @sprintf(
        "dx = %.1f mm, dt = %.2f ms, ten Tusscher 2006 epi — z = %.1f mm mid-plane; white isochrones every 5 ms",
        DX,
        DT,
        Z_SLICE
    );
    fontsize=14,
    color=:gray25,
)
rowgap!(fig.layout, 6)

vidpath = outpath("animation.mp4")
record(fig, vidpath; framerate=20) do io
    return simulate(
        grid,
        f;
        on_frame=(V, act, t) -> begin
            empty!(ax)
            heatmap!(
                ax, xs, ys, view(V, :, :, kz); colorrange=(-90, 40), colormap=:inferno
            )
            contour!(
                ax,
                xs,
                ys,
                view(act, :, :, kz);
                levels=ISO_LEVELS,
                color=(:white, 0.8),
                linewidth=1.3,
            )
            limits!(ax, 0, LX, 0, LY)
            ax.title = @sprintf("t = %.1f ms", t)
            recordframe!(io)
        end,
    )
end
@printf "animation: %s\n" vidpath
