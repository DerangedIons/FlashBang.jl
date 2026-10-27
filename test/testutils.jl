using CytoZoo: CytoZoo

"""
    NullIonicModel()

A one-state cell model whose kinetics are identically zero.

Substituted for a real cell model, it collapses the monodomain equation to pure diffusion,
which has a closed-form solution — so the analytic tests exercise the *whole* pipeline
(`semidiscretize` → dof ranges → `create_initial_condition` → operator splitting →
`getvariable`) against an oracle, rather than poking the diffusion operator directly.

Test-local on purpose: it is a measuring instrument, not a model anyone should reach for.
"""
struct NullIonicModel <: CytoZoo.AbstractCardiacCellModel end

CytoZoo.num_states(::NullIonicModel) = 1
CytoZoo.num_parameters(::NullIonicModel) = 0
CytoZoo.transmembrane_potential_index(::NullIonicModel) = 1
CytoZoo.default_initial_state(::NullIonicModel) = [0.0]
CytoZoo.state_names(::NullIonicModel) = (:v,)
CytoZoo.parameter_names(::NullIonicModel) = ()
CytoZoo.state_index(::NullIonicModel, name::Symbol) = findfirst(==(name), (:v,))
CytoZoo.parameter_index(::NullIonicModel, ::Symbol) = nothing

function (::NullIonicModel)(du, u, p, t)
    du[1] = zero(eltype(du))
    return nothing
end

"""
    GatedNullModel(rate; voltage_index = 1)

`NullIonicModel` plus a gate: a voltage with zero kinetics and a gate `w` decaying as
`dw/dt = -rate·w`, whose `rush_larsen_step!` is the exact exponential `w·exp(-rate·Δt)`.

Two oracles in one instrument. The exact step makes any Rush–Larsen integration of the gate
exact to rounding at every `Δt` — something no right-hand-side integrator can do, so
agreement proves the hook is what ran. And `voltage_index = 2` puts the voltage second, as
gotranx's dependency ordering does, with a gate in block 1 that the diffusion half must
leave alone.
"""
struct GatedNullModel{T} <: CytoZoo.AbstractCardiacCellModel
    rate::T
    voltage_index::Int
end

GatedNullModel(rate; voltage_index::Integer=1) = GatedNullModel(rate, Int(voltage_index))

_gate_index(m::GatedNullModel) = 3 - m.voltage_index

CytoZoo.num_states(::GatedNullModel) = 2
CytoZoo.num_parameters(::GatedNullModel) = 0
CytoZoo.transmembrane_potential_index(m::GatedNullModel) = m.voltage_index
CytoZoo.default_initial_state(::GatedNullModel) = [0.0, 0.0]
CytoZoo.state_names(m::GatedNullModel) = m.voltage_index == 1 ? (:v, :w) : (:w, :v)
CytoZoo.parameter_names(::GatedNullModel) = ()
function CytoZoo.state_index(m::GatedNullModel, name::Symbol)
    return findfirst(==(name), CytoZoo.state_names(m))
end
CytoZoo.parameter_index(::GatedNullModel, ::Symbol) = nothing
CytoZoo.has_rush_larsen(::GatedNullModel) = true

function (m::GatedNullModel)(du, u, p, t)
    du[m.voltage_index] = zero(eltype(du))
    du[_gate_index(m)] = -m.rate * u[_gate_index(m)]
    return nothing
end

function CytoZoo.rush_larsen_step!(u_new, u, p, t, dt, m::GatedNullModel)
    u_new[m.voltage_index] = u[m.voltage_index]
    u_new[_gate_index(m)] = u[_gate_index(m)] * exp(-m.rate * dt)
    return nothing
end

"""
    discrete_decay_rate(κ_eff, grid) -> Real

Decay rate of the lowest cosine mode under the *discrete* zero-flux Laplacian.

MatrixFreeOperators fills Neumann ghosts by symmetric mirroring, and on a cell-centered
grid the mirrored value is exactly the continuum function at the ghost center for
`cos(πx/L)`. The mode is therefore an exact eigenvector of the discrete operator, with
eigenvalue `-(4/Δx²)sin²(πΔx/2L)` per axis instead of the continuum `-(π/L)²`. The two
agree to `O(Δx²)`, and that difference is the entire spatial error budget of the analytic
tests below.
"""
function discrete_decay_rate(κ_eff, grid)
    Δ = spacing(grid)
    extent = [(e[2] - e[1]) for e in grid.extent]
    κ = κ_eff isa Number ? ntuple(_ -> κ_eff, length(Δ)) : κ_eff
    return sum(
        κ[d] * (4 / Δ[d]^2) * sin(π * Δ[d] / (2 * extent[d]))^2 for d in eachindex(Δ)
    )
end

"""
    continuum_decay_rate(κ_eff, grid) -> Real

Decay rate of `∏_d cos(πx_d/L_d)` under the continuum operator `∇⋅κ∇` — the analytic
oracle the pipeline is measured against.
"""
function continuum_decay_rate(κ_eff, grid)
    Δ = spacing(grid)
    extent = [(e[2] - e[1]) for e in grid.extent]
    κ = κ_eff isa Number ? ntuple(_ -> κ_eff, length(Δ)) : κ_eff
    return sum(κ[d] * (π / extent[d])^2 for d in eachindex(Δ))
end

"Lowest zero-flux cosine mode, `∏_d cos(πx_d/L_d)`."
function cosine_mode(grid)
    extent = grid.extent
    return x -> prod(
        cos(π * (x[d] - extent[d][1]) / (extent[d][2] - extent[d][1])) for
        d in eachindex(x)
    )
end

"Run `integrator` to the end of its time span and return the final solution vector."
function run_to_end!(integrator)
    solve!(integrator)
    return integrator.u
end
