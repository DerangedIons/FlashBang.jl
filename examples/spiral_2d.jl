# A reentrant spiral wave on a 2D tissue sheet, from an S1–S2 cross-field protocol.
#
# Aliev–Panfilov kinetics, dimensionless throughout:
#
#   ∂V/∂t = D∇²V − kV(V − a)(V − 1) − VW,    ∂W/∂t = (ε₀ + μ₁W/(μ₂ + V))(−W − kV(V − a − 1))
#
# S1 launches a planar wave from the left edge. S2 fires into the lower-left quadrant
# while that wave is still passing through it: the quadrant's left half has recovered and
# its right half is still refractory, so the new wavefront can only propagate on one side.
# It curls around the free end and locks into a rotating spiral — the mechanism behind
# reentrant arrhythmia, and the reason a wave that should have left the tissue never does.
#
# This is the FlashBang port of MatrixFreeOperators.jl's `examples/monodomain_amr.jl`.
# Two differences worth naming:
#
#   * **Uniform grid, not an adaptive forest.** FlashBang is structured-grid-only by
#     design, so there is no `regrid!` and no cell saving to report. The MFO example
#     exists partly to measure what AMR buys; this one exists to show the physics through
#     FlashBang's API.
#   * **The stimulus is a current, not a state reset.** MFO writes `V = 1` into the
#     stimulated cells directly. Here S1 and S2 are `TransmembraneStimulationProtocol`
#     pulses entering the diffusion half as a source term, which is how FlashBang models
#     stimulation — one protocol, one sign convention, in every dimension.
#
# Run with:   julia --project=examples -t auto examples/spiral_2d.jl
# Finer:      SPIRAL_N=512 SPIRAL_DT=0.005 julia --project=examples -t auto examples/spiral_2d.jl
# Animation:  SPIRAL_MP4=true julia --project=examples -t auto examples/spiral_2d.jl

using Pkg
Pkg.activate(@__DIR__)

using FlashBang
using OrdinaryDiffEqLowOrderRK: Euler
using CairoMakie, Printf, Statistics

include(joinpath(@__DIR__, "aliev_panfilov.jl"))

#--------------------------------------------------------------------------------# Problem

const L = 80.0                     # square sheet, side L (space units)
const N = parse(Int, get(ENV, "SPIRAL_N", "256"))
const DIFF = 1.0                   # χ = Cₘ = 1, so κ is the diffusivity itself
const DT = parse(Float64, get(ENV, "SPIRAL_DT", "0.01"))
const TEND = 220.0

const T_S2 = 50.0                  # S2 timing is the whole experiment: too early and the
const STIM_DUR = 1.0               # quadrant is fully refractory, too late and it has
const STIM_AMP = 1.5               # fully recovered — either way, no wave break.
const S1_WIDTH = 4.0

const SNAP_TIMES = [40.0, 72.0, 110.0, 210.0]
const CAPTIONS = ["planar S1 wave", "S2 wave break", "wavefront curls", "reentrant spiral"]
const SAMPLE_EVERY = 20            # mean(V) samples, for the reentry-period diagnostic

"""
S1 along the left edge at `t = 0`, then S2 into the lower-left quadrant at `T_S2`.

One protocol covers both: `nonzero_intervals` declares the two windows, so outside them
the diffusion right-hand side skips the evaluation entirely rather than calling this
function 65 536 times per step to be told zero.
"""
const STIMULUS = TransmembraneStimulationProtocol(
    function (x, t)
        if t <= STIM_DUR
            return x[1] < S1_WIDTH ? STIM_AMP : 0.0
        elseif T_S2 <= t <= T_S2 + STIM_DUR
            return (x[1] < 0.5L && x[2] < 0.5L) ? STIM_AMP : 0.0
        end
        return 0.0
    end;
    nonzero_intervals=((0.0, STIM_DUR), (T_S2, T_S2 + STIM_DUR)),
)

function build_sheet()
    grid = CartesianGrid(
        ((0.0, L), (0.0, L)), (N, N); bc=ntuple(_ -> (Neumann(), Neumann()), 2)
    )
    model = MonodomainModel(; κ=DIFF, ion=AlievPanfilov(), stim=STIMULUS)
    return grid,
    semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
end

#--------------------------------------------------------------------------------# Solve

"""
Run the S1–S2 protocol to `TEND`, collecting a `V` field at each of `SNAP_TIMES` and a
`mean(V)` trace throughout. `on_frame(V, t)` is called every `frame_every` steps when
given, so the animation reuses this exact solver rather than a parallel copy of it.
"""
function simulate(f; on_frame=nothing, frame_every=120)
    integrator = init(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, TEND)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=DT,
    )

    snaps = NamedTuple[]
    meanV, sample_t = Float64[], Float64[]
    nstep, next_snap = 0, 1

    while integrator.t < TEND - DT / 2
        step!(integrator, DT, true)
        nstep += 1
        φ = getvariable(integrator.u, f, :φₘ)

        if nstep % SAMPLE_EVERY == 0
            push!(meanV, mean(φ))
            push!(sample_t, integrator.t)
        end

        if next_snap <= length(SNAP_TIMES) && integrator.t >= SNAP_TIMES[next_snap]
            push!(snaps, (t=integrator.t, V=reshape(collect(φ), N, N)))
            lo, hi = extrema(φ)
            pct = 100 * count(>(0.5), φ) / length(φ)
            @printf "  t = %6.1f   V ∈ [%6.3f, %5.3f]   depolarized %5.1f%% of the sheet\n" integrator.t lo hi pct
            next_snap += 1
        end

        on_frame !== nothing &&
            nstep % frame_every == 0 &&
            on_frame(reshape(collect(φ), N, N), integrator.t)
    end

    return (; snaps, meanV, sample_t)
end

"""
Dominant reentry period, from the spacing of `mean(V)` maxima after the spiral has
settled.

This is the quantitative test that a spiral actually formed. A wave that failed to break
leaves the tissue and `mean(V)` decays to rest with no further maxima; a rotating spiral
re-excites the sheet once per turn, forever. `NaN` means no reentry.
"""
function reentry_period(meanV, sample_t; t_settle=120.0)
    i0 = findfirst(>=(t_settle), sample_t)
    i0 === nothing && return NaN
    window = i0:length(meanV)
    length(window) < 5 && return NaN
    peaks = [
        i for i in (first(window) + 1):(last(window) - 1) if meanV[i] > meanV[i - 1] &&
            meanV[i] >= meanV[i + 1] &&
            meanV[i] > mean(view(meanV, window))
    ]
    length(peaks) < 2 && return NaN
    return mean(diff(sample_t[peaks]))
end

#--------------------------------------------------------------------------------# Figures

function axes_1d(grid)
    Δ = spacing(grid)[1]
    return range(Δ / 2, L - Δ / 2; length=N)
end

function panel_figure(snaps, xs, path)
    fig = Figure(; size=(1500, 470), fontsize=17)
    hm = nothing
    for (j, s) in enumerate(snaps)
        ax = Axis(
            fig[1, j];
            aspect=DataAspect(),
            title=@sprintf("t = %.0f — %s", s.t, CAPTIONS[j]),
            xlabel="x",
            ylabel=(j == 1 ? "y" : ""),
        )
        hm = heatmap!(ax, xs, xs, s.V; colorrange=(0, 1), colormap=:viridis)
        j > 1 && hideydecorations!(ax; grid=false)
        limits!(ax, 0, L, 0, L)
    end
    ncol = length(snaps) + 1
    Colorbar(fig[1, ncol], hm; label="normalized transmembrane potential V")
    Label(
        fig[0, 1:ncol],
        "Reentrant spiral from an S1–S2 cross-field protocol — FlashBang on a uniform grid";
        fontsize=24,
        font=:bold,
        padding=(0, 0, 2, 0),
    )
    Label(
        fig[2, 1:ncol],
        @sprintf(
            "Aliev–Panfilov kinetics, dimensionless; %d×%d cells over %.0f×%.0f, dt = %.3g. S1 along the left edge, S2 into the lower-left quadrant at t = %.0f.",
            N,
            N,
            L,
            L,
            DT,
            T_S2
        );
        fontsize=15,
        color=:gray25,
    )
    rowgap!(fig.layout, 6)
    save(path, fig)
    return path
end

function trace_figure(result, path)
    fig = Figure(; size=(900, 380), fontsize=16)
    ax = Axis(
        fig[1, 1];
        xlabel="t",
        ylabel="mean V over the sheet",
        title="Reentry is self-sustaining — the sheet keeps re-exciting after the stimuli stop",
    )
    lines!(ax, result.sample_t, result.meanV; linewidth=2.5, color=Makie.wong_colors()[1])
    vlines!(ax, [0.0, T_S2]; color=:firebrick, linestyle=:dash, linewidth=1.5)
    text!(ax, T_S2, maximum(result.meanV); text=" S2", align=(:left, :top), fontsize=15)
    hidespines!(ax, :t, :r)
    save(path, fig)
    return path
end

#--------------------------------------------------------------------------------

grid, f = build_sheet()
xs = axes_1d(grid)

Δx = spacing(grid)[1]
dt_max = Δx^2 / (4 * DIFF)
@printf "S1–S2 spiral: %d×%d cells over %.0f×%.0f, Δx = %.4f, dt = %.4g, to t = %.0f\n" N N L L Δx DT TEND
@printf "  explicit-Euler diffusion limit is dt < %.4g\n" dt_max
DT < dt_max || error("dt = $DT exceeds the explicit stability limit $dt_max for this grid")

if get(ENV, "SPIRAL_MP4", "false") == "true"
    println("recording animation")
    fig = Figure(; size=(620, 620), fontsize=17)
    ax = Axis(fig[1, 1]; aspect=DataAspect(), xlabel="x", ylabel="y")
    Colorbar(fig[1, 2]; colorrange=(0, 1), colormap=:viridis, label="V")
    vidpath = joinpath(@__DIR__, "spiral_2d.mp4")
    record(fig, vidpath; framerate=20) do io
        return simulate(
            f;
            frame_every=120,
            on_frame=(V, t) -> begin
                empty!(ax)
                heatmap!(ax, xs, xs, V; colorrange=(0, 1), colormap=:viridis)
                limits!(ax, 0, L, 0, L)
                ax.title = @sprintf("Aliev–Panfilov spiral — t = %.0f", t)
                recordframe!(io)
            end,
        )
    end
    @printf "animation: %s\n" vidpath
else
    result = simulate(f)
    period = reentry_period(result.meanV, result.sample_t)
    if isnan(period)
        println("  no reentry detected — the S2 wave did not break")
    else
        turns = 100 / period
        @printf "  reentry period %.1f time units (%.3f turns per 100 time units)\n" period turns
    end

    @printf "figure: %s\n" panel_figure(
        result.snaps, xs, joinpath(@__DIR__, "spiral_2d.png")
    )
    @printf "figure: %s\n" trace_figure(result, joinpath(@__DIR__, "spiral_2d_trace.png"))
end
