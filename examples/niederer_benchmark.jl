# The Niederer et al. (2011) N-version cardiac tissue benchmark.
# Phil. Trans. R. Soc. A 369:4331-4351.
#
# Monodomain electrophysiology on a 20 × 7 × 3 mm slab of ventricular tissue with fibres
# along x, ten Tusscher–Panfilov 2006 epicardial kinetics, and a 1.5 mm cube at one corner
# stimulated for 2 ms:
#
#   χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ + Iₛₜᵢₘ),   κ = diag(σ_L, σ_T, σ_T)
#
# The benchmark metric is *activation time* — the first upward crossing of 0 mV — at nine
# fixed points. It is the standard quantitative check for a monodomain solver, because it
# is sensitive to almost everything that can be wrong: the conductivity tensor, the
# anisotropy axis, the stimulus magnitude, the sodium kinetics, and the spatial
# discretization all move it.
#
# This is the FlashBang port of MatrixFreeOperators.jl's `examples/niederer_benchmark.jl`,
# and the two are worth reading side by side. The MFO version is a hand-rolled Godunov
# split over an adaptive block forest, written to measure what AMR buys; this one is the
# same physics expressed through FlashBang's pipeline on a uniform grid, so what it shows
# is the API — one `MonodomainModel`, one `semidiscretize`, one `OperatorSplittingProblem`
# — carrying a real 19-state, three-dimensional, anisotropic problem.
#
# Two consequences of the uniform grid, both expected:
#
#   * every cell is carried at the finest spacing, so this costs roughly what MFO's
#     *uniform control* costs rather than what its adaptive run costs;
#   * cells are cell-centered, so the corner reference points P1…P8 sit half a cell inside
#     the domain rather than exactly on it. That is an O(Δx) offset, and it is the reason
#     the corner activation times are not expected to match the reference to the last
#     digit at coarse spacings.
#
# Run with:      julia --project=examples -t auto examples/niederer_benchmark.jl
# Fast check:    NIEDERER_SMOKE=true  ...   (0.4 mm, 12 ms; seconds, exercises every path)
# Custom:        NIEDERER_DX=0.2 NIEDERER_TEND=60 ...

using Pkg
Pkg.activate(@__DIR__)

using FlashBang
using OrdinaryDiffEqLowOrderRK: Euler
using CairoMakie, LinearAlgebra, Printf

include(joinpath(@__DIR__, "ten_tusscher_2006.jl"))

#--------------------------------------------------------------------------------# Physiology (Niederer 2011 Table 3)

const LX, LY, LZ = 20.0, 7.0, 3.0    # slab dimensions (mm); fibres along x

# Intra- and extracellular conductivities in mS/mm. σ in S/m is numerically equal to
# mS/mm, so the Table 3 values carry over with no conversion factor.
const σ_iL, σ_iT = 0.17, 0.019
const σ_eL, σ_eT = 0.62, 0.24
const σ_L = σ_iL * σ_eL / (σ_iL + σ_eL)   # monodomain harmonic mean, 0.133418 mS/mm
const σ_T = σ_iT * σ_eT / (σ_iT + σ_eT)   #                           0.017606 mS/mm

const BETA = 140.0                   # surface-to-volume ratio (1/mm)
const CM = 0.01                      # membrane capacitance (µF/mm²)

# FlashBang folds κ to the diffusivity κ/(χCₘ) itself, so `κ = (σ_L, σ_T, σ_T)` with
# `χ = BETA` and `Cₘ = CM` reproduces D_L = 0.0953 mm²/ms and D_T = 0.0126 mm²/ms.
const D_L = σ_L / (BETA * CM)
const D_T = σ_T / (BETA * CM)

# 50 000 µA/cm³ ≡ 50 µA/mm³ for 2 ms in a 1.5 mm corner cube. The protocol returns a
# current density per unit membrane area, and FlashBang applies it as Iₛₜᵢₘ/Cₘ — so
# dividing the volumetric density by β gives the 35.7 mV/ms the benchmark specifies.
const STIM_I = 50.0 / BETA           # µA/mm², positive = depolarizing
const STIM_DUR = 2.0                 # ms
const STIM_SIZE = 1.5                # mm
const V_ACTIVATION = 0.0             # mV

#--------------------------------------------------------------------------------# Discretization

const SMOKE = get(ENV, "NIEDERER_SMOKE", "false") == "true"

# 0.1 mm is Niederer's reference resolution and the default. The smoke setting is not a
# cheaper answer — it is a different one — so it is labelled everywhere it is reported.
const DX = parse(Float64, get(ENV, "NIEDERER_DX", SMOKE ? "0.4" : "0.1"))
const TEND = parse(Float64, get(ENV, "NIEDERER_TEND", SMOKE ? "12.0" : "60.0"))
const DT = 0.01                      # ms
const SAMPLE_EVERY = 10              # steps ≡ 0.1 ms, matching the reference quantization

const NCELLS = (round(Int, LX / DX), round(Int, LY / DX), round(Int, LZ / DX))
const SNAP_TIMES = SMOKE ? [4.0, 8.0, 11.0] : [5.0, 15.0, 30.0, 45.0]
const Z_SLICE = 1.5                  # mid-plane for the figures (mm)
const NDIAG = 50                     # samples along the P1 → P8 diagonal

outpath(name) = joinpath(@__DIR__, SMOKE ? "niederer_smoke_$name" : "niederer_$name")

#--------------------------------------------------------------------------------# Reference points

# Niederer 2011 Figure 1b: the eight slab corners plus the centre. P1 is the stimulated
# corner and P8 the far corner, the first and last to activate.
const P_NAMES = ("P1", "P2", "P3", "P4", "P5", "P6", "P7", "P8", "P9")
const P_REF = (
    (0.0, 0.0, 0.0),
    (0.0, 7.0, 0.0),
    (20.0, 7.0, 0.0),
    (20.0, 0.0, 0.0),
    (0.0, 0.0, 3.0),
    (0.0, 7.0, 3.0),
    (20.0, 0.0, 3.0),
    (20.0, 7.0, 3.0),
    (10.0, 3.5, 1.5),
)

# Cross-code reference: a structured 7-point finite-difference solve of the same problem
# at dx = 0.1 mm, dt = 0.01 ms (RadialBasisFunctions.jl/Macchiato). An independent
# discretization, so agreement validates the physics rather than just the grid.
const FD_REFERENCE = (1.3, 30.1, 42.9, 31.9, 9.1, 31.5, 33.0, 43.8, 19.9)

#--------------------------------------------------------------------------------# Problem

"""
The stimulus: a 1.5 mm cube at the origin corner, on for the first 2 ms.

`nonzero_intervals` matters here — without it the diffusion right-hand side would evaluate
this predicate at every one of 420 000 nodes on every one of 6 000 steps, to be told zero
on 5 800 of them.
"""
const STIMULUS = TransmembraneStimulationProtocol(
    (x, t) -> (x[1] <= STIM_SIZE && x[2] <= STIM_SIZE && x[3] <= STIM_SIZE) ? STIM_I : 0.0;
    nonzero_intervals=((0.0, STIM_DUR),),
)

function build_slab()
    grid = CartesianGrid(
        ((0.0, LX), (0.0, LY), (0.0, LZ)), NCELLS; bc=ntuple(_ -> (Neumann(), Neumann()), 3)
    )
    # Axis-aligned diagonal anisotropy: FlashBang assembles this as
    # D_T∇² + (D_L − D_T)∂ₓₓ, one halo exchange cheaper than three second derivatives,
    # and exact because κ is diagonal. Zero-flux is the correct boundary for it:
    # n·κ∇φ = 0 reduces to ∂φ/∂n = 0 when κ is axis-aligned.
    model = MonodomainModel(;
        κ=(σ_L, σ_T, σ_T), χ=BETA, Cₘ=CM, ion=TenTusscherEpi(), stim=STIMULUS
    )
    return grid,
    semidiscretize(ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid)
end

#--------------------------------------------------------------------------------# Probing

"""
    node_index(grid, x) -> Int

Linear index of the cell whose interior contains `x`, in the same ordering
[`node_coordinates`](@ref) uses.

Computed from the spacing rather than searched: on a uniform grid the containing cell is
arithmetic, and a 420 000-node nearest-neighbour scan per probe point is not.
"""
function node_index(grid, x)
    Δ, n = spacing(grid), local_size(grid)
    idx = ntuple(
        d -> clamp(ceil(Int, (x[d] - grid.extent[d][1]) / Δ[d]), 1, n[d]), length(n)
    )
    return LinearIndices(n)[idx...]
end

"Points evenly spaced along the P1 → P8 diagonal, and their arc lengths."
function diagonal_points()
    p1, p8 = collect(P_REF[1]), collect(P_REF[8])
    total = norm(p8 - p1)
    ss = range(0, 1; length=NDIAG)
    return [Tuple(p1 + s * (p8 - p1)) for s in ss], [s * total for s in ss]
end

#--------------------------------------------------------------------------------# Solve

function simulate(grid, f)
    integrator = init(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, TEND)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=DT,
    )

    diagpts, diagdist = diagonal_points()
    probes = [node_index(grid, x) for x in vcat(collect(P_REF), diagpts)]
    activation = fill(NaN, length(probes))

    snaps = NamedTuple[]
    nstep, next_snap = 0, 1
    tstart = time_ns()

    while integrator.t < TEND - DT / 2
        step!(integrator, DT, true)
        nstep += 1

        if nstep % SAMPLE_EVERY == 0
            φ = getvariable(integrator.u, f, :φₘ)
            @inbounds for (k, i) in enumerate(probes)
                isnan(activation[k]) &&
                    φ[i] > V_ACTIVATION &&
                    (activation[k] = integrator.t)
            end
        end

        if next_snap <= length(SNAP_TIMES) && integrator.t >= SNAP_TIMES[next_snap]
            φ = collect(getvariable(integrator.u, f, :φₘ))
            push!(snaps, (t=integrator.t, V=reshape(φ, NCELLS...)))
            lo, hi = extrema(φ)
            pct = 100 * count(>(V_ACTIVATION), φ) / length(φ)
            @printf "  t = %5.1f ms   φₘ ∈ [%7.2f, %6.2f] mV   %5.1f%% activated\n" integrator.t lo hi pct
            next_snap += 1
        end
    end

    wall = (time_ns() - tstart) * 1.0e-9
    cellsteps = prod(NCELLS) * nstep
    ms_per_step = 1.0e3 * wall / nstep
    ns_per_update = 1.0e9 * wall / cellsteps
    @printf "  %d steps in %.1f s (%.2f ms/step, %.1f ns per cell-update)\n" nstep wall ms_per_step ns_per_update

    return (; act_ref=activation[1:9], act_diag=activation[10:end], diagdist, snaps, wall)
end

#--------------------------------------------------------------------------------# Reporting

function report(act_ref)
    @printf "\n  %-4s %-22s %12s %12s %9s\n" "" "target (mm)" "FlashBang" "FD ref" "Δ"
    for k in 1:9
        p = P_REF[k]
        a = act_ref[k]
        astr = isnan(a) ? "  not activ." : @sprintf("%9.1f ms", a)
        dstr = isnan(a) ? "     —" : @sprintf("%+6.1f", a - FD_REFERENCE[k])
        @printf "  %-4s (%5.1f,%4.1f,%4.1f) %s %9.1f ms %s\n" P_NAMES[k] p[1] p[2] p[3] astr FD_REFERENCE[k] dstr
    end
    good = [k for k in 1:9 if !isnan(act_ref[k])]
    if !isempty(good)
        err = [abs(act_ref[k] - FD_REFERENCE[k]) for k in good]
        mean_err, max_err = sum(err) / length(err), maximum(err)
        @printf "\n  %d/9 activated;  mean |Δ| vs FD %.2f ms,  max %.2f ms\n" length(good) mean_err max_err
    end
    return nothing
end

#--------------------------------------------------------------------------------# Figures

function slice_axes(grid)
    Δ = spacing(grid)
    return (
        range(Δ[1] / 2, LX - Δ[1] / 2; length=NCELLS[1]),
        range(Δ[2] / 2, LY - Δ[2] / 2; length=NCELLS[2]),
    )
end

function slices_figure(snaps, grid, path)
    xs, ys = slice_axes(grid)
    k = clamp(ceil(Int, Z_SLICE / spacing(grid)[3]), 1, NCELLS[3])

    fig = Figure(; size=(1180, 210 * length(snaps) + 150), fontsize=17)
    hm = nothing
    for (j, s) in enumerate(snaps)
        ax = Axis(
            fig[j, 1];
            aspect=DataAspect(),
            ylabel="y (mm)",
            xlabel=(j == length(snaps) ? "x (mm)" : ""),
            title=@sprintf("t = %.0f ms", s.t),
        )
        hm = heatmap!(
            ax, xs, ys, view(s.V, :, :, k); colorrange=(-90, 40), colormap=:inferno
        )
        j < length(snaps) && hidexdecorations!(ax; grid=false)
        limits!(ax, 0, LX, 0, LY)
    end
    Colorbar(fig[1:length(snaps), 2], hm; label="transmembrane potential φₘ (mV)")
    Label(
        fig[0, 1:2],
        @sprintf("Niederer benchmark in FlashBang — z = %.1f mm mid-plane", Z_SLICE);
        fontsize=23,
        font=:bold,
    )
    Label(
        fig[length(snaps) + 1, 1:2],
        @sprintf(
            "ten Tusscher–Panfilov 2006 epicardial; D = diag(%.4f, %.4f, %.4f) mm²/ms, fibres along x; %d×%d×%d cells at Δx = %.2f mm.",
            D_L,
            D_T,
            D_T,
            NCELLS[1],
            NCELLS[2],
            NCELLS[3],
            DX
        );
        fontsize=14,
        color=:gray25,
    )
    rowgap!(fig.layout, 6)
    save(path, fig)
    return path
end

function activation_figure(dist, at_diag, path)
    fig = Figure(; size=(820, 480), fontsize=17)
    ax = Axis(
        fig[1, 1];
        xlabel="distance along the P1 → P8 diagonal (mm)",
        ylabel="activation time (ms)",
        title="Activation along the slab diagonal",
    )
    lines!(ax, dist, at_diag; color=Makie.wong_colors()[1], linewidth=3, label="FlashBang")

    # Only the reference points that actually lie on this diagonal belong here — P1, the
    # centre P9, and the far corner P8. The other six are corners off the line; they are
    # in the printed table, not on this plot.
    p1, p8 = collect(P_REF[1]), collect(P_REF[8])
    û = (p8 - p1) / norm(p8 - p1)
    on_diag = [
        k for k in eachindex(P_REF) if
        norm((collect(P_REF[k]) - p1) - ((collect(P_REF[k]) - p1) ⋅ û) * û) < 1.0e-9
    ]
    scatter!(
        ax,
        [norm(collect(P_REF[k]) - p1) for k in on_diag],
        [FD_REFERENCE[k] for k in on_diag];
        color=:firebrick,
        markersize=14,
        marker=:diamond,
        label="finite-difference reference, Δx = 0.1 mm",
    )
    axislegend(ax; position=:lt, framevisible=false)
    save(path, fig)
    return path
end

function write_diagonal_csv(dist, at, path)
    pts, _ = diagonal_points()
    open(path, "w") do io
        println(io, "distance_mm,x_mm,y_mm,z_mm,activation_time_ms")
        for i in eachindex(dist)
            a = isnan(at[i]) ? "NaN" : @sprintf("%.2f", at[i])
            @printf io "%.4f,%.4f,%.4f,%.4f,%s\n" dist[i] pts[i][1] pts[i][2] pts[i][3] a
        end
    end
    return path
end

#--------------------------------------------------------------------------------

grid, f = build_slab()

@printf "Niederer 2011 benchmark in FlashBang — %s\n" (
    SMOKE ? "SMOKE mode (coarse and short; not the benchmark answer)" : "full run"
)
@printf "  slab %.0f × %.0f × %.0f mm, D = (%.5f, %.5f, %.5f) mm²/ms, fibres along x\n" LX LY LZ D_L D_T D_T
ncell = prod(NCELLS)
nstates = solution_size(f) ÷ num_nodes(f)
gigabytes = 8.0e-9 * solution_size(f)
@printf "  %d × %d × %d = %d cells at Δx = %.3f mm, %d states each (%.2f GB of solution vector)\n" NCELLS[1] NCELLS[2] NCELLS[3] ncell DX nstates gigabytes
@printf "  dt = %.3f ms to t = %.0f ms, %d threads\n\n" DT TEND Threads.nthreads()

# The diffusion half is explicit Euler, so the spacing sets a hard step limit. Fail loudly
# rather than after twenty minutes of NaNs.
Δ = spacing(grid)
dt_max = 1 / (2 * (D_L / Δ[1]^2 + D_T / Δ[2]^2 + D_T / Δ[3]^2))
@printf "  explicit-Euler diffusion limit is dt < %.4f ms\n\n" dt_max
DT < dt_max || error("dt = $DT exceeds the explicit stability limit $dt_max for Δx = $DX")

result = simulate(grid, f)
report(result.act_ref)

@printf "\nfigure: %s\n" slices_figure(result.snaps, grid, outpath("benchmark.png"))
@printf "figure: %s\n" activation_figure(
    result.diagdist, result.act_diag, outpath("activation.png")
)
@printf "data:   %s\n" write_diagonal_csv(
    result.diagdist, result.act_diag, outpath("diagonal.csv")
)
