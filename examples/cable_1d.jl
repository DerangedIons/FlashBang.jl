# A propagating action potential on a one-dimensional cable — the smallest complete
# FlashBang problem, and the shape every larger one keeps.
#
#   χCₘ∂ₜφₘ = ∇⋅κ∇φₘ + χ(Iᵢₒₙ + Iₛₜᵢₘ),   ∂ₜs = f(φₘ, s)
#
# with FitzHugh–Nagumo kinetics, zero-flux ends, and a brief depolarizing pulse at the left
# end. The pulse pushes the leftmost nodes past FHN's excitation threshold; diffusion carries
# that above threshold at the next nodes, which fire in turn, and the result is a wave
# travelling at a speed set by the kinetics and κ rather than by anything imposed.
#
# The five lines that matter are the pipeline itself:
#
#   model = MonodomainModel(; κ, ion, stim)                    the continuous problem
#   split = ReactionDiffusionSplit(model)                      how to split it
#   f     = semidiscretize(split, FiniteDifferenceDiscretization(), grid)
#   prob  = OperatorSplittingProblem(f, create_initial_condition(f), tspan)
#   sol   = solve(prob, LieTrotterGodunov((Euler(), Euler())); dt, saveat)
#
# Run with:  julia --project=examples -t auto examples/cable_1d.jl

using Pkg
Pkg.activate(@__DIR__)

using FlashBang
using CytoZoo: FHNModel
using OrdinaryDiffEqLowOrderRK: Euler
using CairoMakie, Printf

#--------------------------------------------------------------------------------# Problem

const L = 20.0                # cable length
const NCELLS = 400
const κ = 0.1                 # conductivity; χ = Cₘ = 1, so the diffusivity is κ itself
const DT = 0.005
const TEND = 400.0

const STIM_AMPLITUDE = 0.5    # positive is depolarizing — see AbstractStimulationProtocol
const STIM_WIDTH = 1.0
const STIM_DURATION = 2.0
const THRESHOLD = 0.5         # φₘ runs over roughly [0, 1]; halfway marks activation
const SAMPLE_EVERY = 20       # store φₘ every 0.1 time units for the space-time map

"""
Assemble the cable: zero-flux grid, FHN kinetics, and a brief pulse at the left end.

`nonzero_intervals` lets the diffusion right-hand side skip the stimulus evaluation
entirely once the pulse is over, which is 99.5% of this run. FHN's own stimulus stays at
its zero default: in tissue the stimulus belongs to the diffusion half, and the two sign
conventions are opposite.
"""
function build_cable()
    grid = CartesianGrid(((0.0, L),), (NCELLS,); bc=((Neumann(), Neumann()),))
    stim = TransmembraneStimulationProtocol(
        (x, t) -> (t <= STIM_DURATION && x[1] <= STIM_WIDTH) ? STIM_AMPLITUDE : 0.0;
        nonzero_intervals=((0.0, STIM_DURATION),),
    )
    model = MonodomainModel(; κ, ion=FHNModel(), stim)
    return semidiscretize(
        ReactionDiffusionSplit(model), FiniteDifferenceDiscretization(), grid
    )
end

"""
Solve the cable to `TEND`, saving φₘ every `SAMPLE_EVERY` steps for the space-time map and
recording the first threshold crossing at every node.

The snapshots are ordinary `saveat` points. Activation needs every step, which would be far
too many full-state saves, so it is observed in flight by a [`StateObserver`](@ref)
instead. `save_start = false` leaves the t = 0 frame out of the map, as before.
"""
function run_cable(f)
    activation = fill(NaN, NCELLS)
    record = StateObserver() do u, t
        φ = getvariable(u, f, :φₘ)
        @inbounds for i in eachindex(activation)
            isnan(activation[i]) && φ[i] > THRESHOLD && (activation[i] = t)
        end
    end

    sol = solve(
        OperatorSplittingProblem(f, create_initial_condition(f), (0.0, TEND)),
        LieTrotterGodunov((Euler(), Euler()));
        dt=DT,
        saveat=SAMPLE_EVERY * DT,
        save_start=false,
        callback=record,
    )

    return (; activation, frames=sol[:φₘ], times=sol.t, φ_final=sol[:φₘ, end])
end

#--------------------------------------------------------------------------------# Reporting

function report(result, xs)
    activated = count(!isnan, result.activation)
    @printf "  %d/%d nodes activated\n" activated NCELLS
    if activated == NCELLS
        cv = (xs[end] - xs[1]) / (result.activation[end] - result.activation[1])
        @printf "  activation %.2f → %.2f, conduction velocity %.4f length units per time unit\n" result.activation[1] result.activation[end] cv
        @printf "  monotone in x: %s\n" all(>=(0), diff(result.activation))
    end
    lo, hi = extrema(result.φ_final)
    @printf "  final φₘ ∈ [%.4f, %.4f]\n" lo hi
    return nothing
end

function figure(result, xs, path)
    fig = Figure(; size=(1000, 760), fontsize=16)

    ax1 = Axis(
        fig[1, 1];
        xlabel="x",
        ylabel="t",
        title="Transmembrane potential φₘ(x, t)",
        subtitle="the diagonal band is the wavefront; its slope is 1/CV",
    )
    hm = heatmap!(
        ax1,
        xs,
        result.times,
        reduce(hcat, result.frames);
        colormap=:inferno,
        colorrange=(0, 1),
    )
    Colorbar(fig[1, 2], hm; label="φₘ")

    ax2 = Axis(
        fig[2, 1];
        xlabel="x",
        ylabel="activation time",
        title="Activation time — straight means a steadily propagating wave",
    )
    lines!(ax2, xs, result.activation; linewidth=3, color=Makie.wong_colors()[1])

    ax3 = Axis(fig[3, 1]; xlabel="x", ylabel="φₘ", title="Snapshots")
    for k in round.(Int, range(1, length(result.frames); length=6))
        lines!(ax3, xs, result.frames[k]; label=@sprintf("t = %.0f", result.times[k]))
    end
    axislegend(ax3; position=:rt, framevisible=false, nbanks=3)

    Label(
        fig[0, 1:2],
        @sprintf(
            "FitzHugh–Nagumo cable — κ = %.3g, zero flux, %.1f-wide pulse of amplitude %.2f for %.0f time units",
            κ,
            STIM_WIDTH,
            STIM_AMPLITUDE,
            STIM_DURATION
        );
        fontsize=18,
        font=:bold,
    )
    rowgap!(fig.layout, 8)

    save(path, fig)
    return path
end

#--------------------------------------------------------------------------------

f = build_cable()
xs = [x[1] for x in node_coordinates(f)]

Δx = L / NCELLS
nstates = solution_size(f) ÷ num_nodes(f)
@printf "1D FHN cable: %d nodes over %.0f length units, Δx = %.4f, dt = %.4f, %d states\n" NCELLS L Δx DT nstates

result = run_cable(f)
report(result, xs)
@printf "figure: %s\n" figure(result, xs, joinpath(@__DIR__, "cable_1d.png"))
