# ten Tusscher & Panfilov (2006) epicardial human ventricular cell model, as a CytoZoo
# cell model. "Alternans and spiral breakup in a human ventricular tissue model",
# Am J Physiol Heart Circ Physiol 291:H1088-H1100.
#
# The ionic model the Niederer 2011 benchmark is defined with (see
# niederer_benchmark.jl) — the published reference activation times are TT06-specific,
# so substituting another model would turn the benchmark into an unvalidated slab run.
#
# The kinetics below are a verbatim copy of the validated port in
# MatrixFreeOperators.jl's `examples/ten_tusscher_2006.jl`; what is new here is the
# CytoZoo interface wrapped around it. That interface flattens the layout: MFO kept `V`
# out of the state vector because its tissue solver carried voltage in its own field,
# whereas a CytoZoo model owns all 19 states and FlashBang's state-blocked layout gives
# the diffusion half the leading `V` block for free.
#
# Float64 only. The published constants are `Float64` literals and are left that way —
# `Float32` monodomain is not a supported configuration (see the FlashBang README).
#
# Lives in `examples/` rather than in CytoZoo because it is here to drive a benchmark.
# Promoting it to the zoo is a separate conversation.
#
# Run standalone to verify the port against single-cell targets:
#   julia --project=examples examples/ten_tusscher_2006.jl

using Pkg
Pkg.activate(@__DIR__)

module TenTusscher2006

using CytoZoo: CytoZoo, SpatialContext, Stimulus
using StaticArrays: SVector

export TenTusscherEpi, NSTATES, initial_voltage, initial_state, rhs

#--------------------------------------------------------------------------------# State layout

"""
Number of gating and concentration states per cell, *excluding* the transmembrane
potential. The CytoZoo state vector is `[V; these]`, so it has `NSTATES + 1` entries.

Order: `Xr1 Xr2 Xs m h j d f f2 fCass s r Ca_i Ca_SR Ca_ss R′ Na_i K_i`.
"""
const NSTATES = 18

const IXR1, IXR2, IXS, IM, IH, IJ, ID, IF, IF2, IFCASS = 1, 2, 3, 4, 5, 6, 7, 8, 9, 10
const IS, IR, ICAI, ICASR, ICASS, IRPRIME, INAI, IKI = 11, 12, 13, 14, 15, 16, 17, 18

#--------------------------------------------------------------------------------# Parameters

# Physical constants
const R_GAS = 8314.472        # mJ/(mol·K)
const TEMP = 310.0            # K
const FARADAY = 96485.3415    # C/mol
const RTOF = R_GAS * TEMP / FARADAY
const FORT = 1.0 / RTOF

# Cell geometry
const CM_CELL = 0.185         # µF/cm² — appears only in the concentration conversions
const V_C = 16404.0           # µm³
const V_SR = 1094.0           # µm³
const V_SS = 54.68            # µm³

# External concentrations (mM)
const K_O = 5.4
const NA_O = 140.0
const CA_O = 2.0

# Maximum conductances — epicardial variant (G_to and G_Ks are what make it epicardial)
const G_NA = 14.838
const G_BNA = 0.00029
const G_CAL = 3.98e-5
const G_BCA = 0.000592
const G_TO = 0.294
const G_KS = 0.392
const G_KR = 0.153
const G_K1 = 5.405
const G_PCA = 0.1238
const G_PK = 0.0146
const P_KNA = 0.03

# Pump and exchanger
const P_NAK = 2.724
const K_MK = 1.0
const K_MNA = 40.0
const K_NACA = 1000.0
const K_SAT = 0.1
const ALPHA_NACA = 2.5
const GAMMA_NACA = 0.35
const KM_CA = 1.38
const KM_NAI = 87.5
const K_PCA = 0.0005

# Calcium dynamics
const K1_PRIME = 0.15
const K2_PRIME = 0.045
const K3 = 0.06
const K4 = 0.005
const EC = 1.5
const MAX_SR = 2.5
const MIN_SR = 1.0
const V_REL = 0.102
const V_XFER = 0.0038
const K_UP = 0.00025
const V_LEAK = 0.00036
const VMAX_UP = 0.006375

# Buffering
const BUF_C = 0.2
const K_BUF_C = 0.001
const BUF_SR = 10.0
const K_BUF_SR = 0.3
const BUF_SS = 0.4
const K_BUF_SS = 0.00025

#--------------------------------------------------------------------------------# Initial conditions

"Resting transmembrane potential (mV), Niederer 2011 Table 2."
initial_voltage() = -85.23

"""
    initial_state() -> SVector{18,Float64}

Resting gating and concentration states, Niederer 2011 Table 2.
"""
function initial_state()
    return SVector{NSTATES,Float64}(
        0.00621,    # Xr1
        0.4712,     # Xr2
        0.0095,     # Xs
        0.00172,    # m
        0.7444,     # h
        0.7045,     # j
        3.373e-5,   # d
        0.7888,     # f
        0.9755,     # f2
        0.9953,     # fCass
        0.999998,   # s
        2.42e-8,    # r
        0.000126,   # Ca_i  (mM)
        3.64,       # Ca_SR (mM)
        0.00036,    # Ca_ss (mM)
        0.9073,     # R′
        8.604,      # Na_i  (mM)
        136.89,     # K_i   (mM)
    )
end

#--------------------------------------------------------------------------------# Reaction kernel

"""
    rhs(V, s::SVector{18}, stim) -> (dV, ds)

Time derivatives of the transmembrane potential and the 18 cell states. `stim` is
the stimulus contribution to `dV/dt` in mV/ms (zero outside the stimulated region
or window) — currents are in pA/pF ≡ mV/ms, so `dV/dt = -I_ion + stim` with no
capacitance division. `CM_CELL` enters only where currents drive concentrations.

Gates are clamped to `[0,1]` and concentrations to `≥ 1e-10` on read. This is not
defensive tidying: forward Euler on the fast gates undershoots by O(1e-10) per
step and the downstream `log` then throws, and under AMR a cell created by
refinement can inherit a marginally out-of-range gate from the one-sided
`(5/4, -1/4)` prolongation weights at a parent-block edge.
"""
@inline function rhs(V, s::SVector{NSTATES}, stim)
    Xr1 = clamp(s[IXR1], 0.0, 1.0)
    Xr2 = clamp(s[IXR2], 0.0, 1.0)
    Xs = clamp(s[IXS], 0.0, 1.0)
    m = clamp(s[IM], 0.0, 1.0)
    h = clamp(s[IH], 0.0, 1.0)
    j = clamp(s[IJ], 0.0, 1.0)
    d = clamp(s[ID], 0.0, 1.0)
    f = clamp(s[IF], 0.0, 1.0)
    f2 = clamp(s[IF2], 0.0, 1.0)
    fCass = clamp(s[IFCASS], 0.0, 1.0)
    sg = clamp(s[IS], 0.0, 1.0)
    r = clamp(s[IR], 0.0, 1.0)
    Ca_i = max(s[ICAI], 1.0e-10)
    Ca_SR = max(s[ICASR], 1.0e-10)
    Ca_ss = max(s[ICASS], 1.0e-10)
    Rp = clamp(s[IRPRIME], 0.0, 1.0)
    Na_i = max(s[INAI], 1.0e-10)
    K_i = max(s[IKI], 1.0e-10)

    # Reversal potentials
    E_Na = RTOF * log(NA_O / Na_i)
    E_K = RTOF * log(K_O / K_i)
    E_Ks = RTOF * log((K_O + P_KNA * NA_O) / (K_i + P_KNA * Na_i))
    E_Ca = 0.5 * RTOF * log(CA_O / Ca_i)

    # Fast and background sodium
    I_Na = G_NA * m^3 * h * j * (V - E_Na)
    I_bNa = G_BNA * (V - E_Na)

    # L-type calcium, Goldman-Hodgkin-Katz. The driving term is singular at
    # V = 15 mV; take the L'Hôpital limit Veff/(exp(2·Veff·FORT) - 1) → 1/(2·FORT).
    Veff = V - 15.0
    I_CaL = if abs(Veff) < 1.0e-6
        G_CAL * d * f * f2 * fCass * 2.0 * FARADAY *
        (0.25 * Ca_ss * exp(2.0 * Veff * FORT) - CA_O)
    else
        e2v = exp(2.0 * Veff * FORT)
        G_CAL * d * f * f2 * fCass * 4.0 * Veff * FARADAY^2 / (R_GAS * TEMP) *
        (0.25 * Ca_ss * e2v - CA_O) / (e2v - 1.0)
    end

    I_bCa = G_BCA * (V - E_Ca)
    I_to = G_TO * r * sg * (V - E_K)
    I_Ks = G_KS * Xs^2 * (V - E_Ks)
    I_Kr = G_KR * sqrt(K_O / 5.4) * Xr1 * Xr2 * (V - E_K)

    # Inward rectifier
    a_K1 = 0.1 / (1.0 + exp(0.06 * (V - E_K - 200.0)))
    b_K1 = (3.0 * exp(0.0002 * (V - E_K + 100.0)) + exp(0.1 * (V - E_K - 10.0))) /
           (1.0 + exp(-0.5 * (V - E_K)))
    I_K1 = G_K1 * (a_K1 / (a_K1 + b_K1)) * (V - E_K)

    # Sodium-potassium pump
    Vfrt = V * FORT
    I_NaK = P_NAK * K_O / (K_O + K_MK) * Na_i / (Na_i + K_MNA) /
            (1.0 + 0.1245 * exp(-0.1 * Vfrt) + 0.0353 * exp(-Vfrt))

    # Sodium-calcium exchanger
    eg = exp(GAMMA_NACA * Vfrt)
    egm1 = exp((GAMMA_NACA - 1.0) * Vfrt)
    I_NaCa = K_NACA * (eg * Na_i^3 * CA_O - egm1 * NA_O^3 * Ca_i * ALPHA_NACA) /
             ((KM_NAI^3 + NA_O^3) * (KM_CA + CA_O) * (1.0 + K_SAT * egm1))

    I_pCa = G_PCA * Ca_i / (K_PCA + Ca_i)
    I_pK = G_PK * (V - E_K) / (1.0 + exp((25.0 - V) / 5.98))

    I_ion = I_Na + I_bNa + I_CaL + I_bCa + I_to + I_Ks + I_Kr + I_K1 +
            I_NaK + I_NaCa + I_pCa + I_pK

    # Gating kinetics, all in dx/dt = (x_inf - x)/tau_x form
    xr1_inf = 1.0 / (1.0 + exp((-26.0 - V) / 7.0))
    tau_xr1 = (450.0 / (1.0 + exp((-45.0 - V) / 10.0))) *
              (6.0 / (1.0 + exp((V + 30.0) / 11.5)))
    dXr1 = (xr1_inf - Xr1) / tau_xr1

    xr2_inf = 1.0 / (1.0 + exp((V + 88.0) / 24.0))
    tau_xr2 = (3.0 / (1.0 + exp((-60.0 - V) / 20.0))) *
              (1.12 / (1.0 + exp((V - 60.0) / 20.0)))
    dXr2 = (xr2_inf - Xr2) / tau_xr2

    xs_inf = 1.0 / (1.0 + exp((-5.0 - V) / 14.0))
    tau_xs = (1400.0 / sqrt(1.0 + exp((5.0 - V) / 6.0))) *
             (1.0 / (1.0 + exp((V - 35.0) / 15.0))) + 80.0
    dXs = (xs_inf - Xs) / tau_xs

    m_inf = (1.0 / (1.0 + exp((-56.86 - V) / 9.03)))^2
    tau_m = (1.0 / (1.0 + exp((-60.0 - V) / 5.0))) *
            (0.1 / (1.0 + exp((V + 35.0) / 5.0)) + 0.1 / (1.0 + exp((V - 50.0) / 200.0)))
    dm = (m_inf - m) / tau_m

    h_inf = (1.0 / (1.0 + exp((V + 71.55) / 7.43)))^2
    a_h, b_h = if V >= -40.0
        (0.0, 0.77 / (0.13 * (1.0 + exp(-(V + 10.66) / 11.1))))
    else
        (0.057 * exp(-(V + 80.0) / 6.8), 2.7 * exp(0.079 * V) + 3.1e5 * exp(0.3485 * V))
    end
    dh = (h_inf - h) * (a_h + b_h)

    a_j, b_j = if V >= -40.0
        (0.0, 0.6 * exp(0.057 * V) / (1.0 + exp(-0.1 * (V + 32.0))))
    else
        ((-2.5428e4 * exp(0.2444 * V) - 6.948e-6 * exp(-0.04391 * V)) *
         (V + 37.78) / (1.0 + exp(0.311 * (V + 79.23))),
         0.02424 * exp(-0.01052 * V) / (1.0 + exp(-0.1378 * (V + 40.14))))
    end
    dj = (h_inf - j) * (a_j + b_j)      # j_inf == h_inf

    d_inf = 1.0 / (1.0 + exp((-8.0 - V) / 7.5))
    tau_d = (1.4 / (1.0 + exp((-35.0 - V) / 13.0)) + 0.25) *
            (1.4 / (1.0 + exp((V + 5.0) / 5.0))) + 1.0 / (1.0 + exp((50.0 - V) / 20.0))
    dd = (d_inf - d) / tau_d

    f_inf = 1.0 / (1.0 + exp((V + 20.0) / 7.0))
    tau_f = 1102.5 * exp(-(V + 27.0)^2 / 225.0) + 200.0 / (1.0 + exp((13.0 - V) / 10.0)) +
            180.0 / (1.0 + exp((V + 30.0) / 10.0)) + 20.0
    df = (f_inf - f) / tau_f

    f2_inf = 0.67 / (1.0 + exp((V + 35.0) / 7.0)) + 0.33
    tau_f2 = 562.0 * exp(-(V + 27.0)^2 / 240.0) + 31.0 / (1.0 + exp((25.0 - V) / 10.0)) +
             80.0 / (1.0 + exp((V + 30.0) / 10.0))
    df2 = (f2_inf - f2) / tau_f2

    ca_ratio = (Ca_ss / 0.05)^2
    fCass_inf = 0.6 / (1.0 + ca_ratio) + 0.4
    tau_fCass = 80.0 / (1.0 + ca_ratio) + 2.0
    dfCass = (fCass_inf - fCass) / tau_fCass

    s_inf = 1.0 / (1.0 + exp((V + 20.0) / 5.0))                    # epicardial
    tau_s = 85.0 * exp(-(V + 45.0)^2 / 320.0) + 5.0 / (1.0 + exp((V - 20.0) / 5.0)) + 3.0
    ds = (s_inf - sg) / tau_s

    r_inf = 1.0 / (1.0 + exp((20.0 - V) / 6.0))
    tau_r = 9.5 * exp(-(V + 40.0)^2 / 1800.0) + 0.8
    dr = (r_inf - r) / tau_r

    # Ryanodine receptor and SR fluxes
    kcasr = MAX_SR - (MAX_SR - MIN_SR) / (1.0 + (EC / Ca_SR)^2)
    k1 = K1_PRIME / kcasr
    k2 = K2_PRIME * kcasr
    k1cass2 = k1 * Ca_ss^2
    O = k1cass2 * Rp / (K3 + k1cass2)
    dRp = -k2 * Ca_ss * Rp + K4 * (1.0 - Rp)

    I_rel = V_REL * O * (Ca_SR - Ca_ss)
    I_up = VMAX_UP / (1.0 + K_UP^2 / Ca_i^2)
    I_leak = V_LEAK * (Ca_SR - Ca_i)
    I_xfer = V_XFER * (Ca_ss - Ca_i)

    # Buffering
    bufc = 1.0 / (1.0 + BUF_C * K_BUF_C / (Ca_i + K_BUF_C)^2)
    bufsr = 1.0 / (1.0 + BUF_SR * K_BUF_SR / (Ca_SR + K_BUF_SR)^2)
    bufss = 1.0 / (1.0 + BUF_SS * K_BUF_SS / (Ca_ss + K_BUF_SS)^2)

    # I_up, I_leak and I_rel are referenced to the SR volume; I_xfer to the cytosol.
    # Hence the V_SR/V_C on the SR fluxes entering the cytosol, none on dCa_SR, and
    # the matching V_SR/V_SS and V_C/V_SS below. Getting these ratios wrong still
    # conserves total calcium, so it shows up only as a crushed Ca transient.
    dCa_i = bufc * ((I_leak - I_up) * V_SR / V_C + I_xfer -
                    (I_bCa + I_pCa - 2.0 * I_NaCa) * CM_CELL / (2.0 * V_C * FARADAY) * 1.0e6)
    dCa_SR = bufsr * (I_up - I_rel - I_leak)
    dCa_ss = bufss * (-I_CaL * CM_CELL / (2.0 * V_SS * FARADAY) * 1.0e6 +
                      I_rel * V_SR / V_SS - I_xfer * V_C / V_SS)

    dNa_i = -(I_Na + I_bNa + 3.0 * I_NaK + 3.0 * I_NaCa) * CM_CELL / (V_C * FARADAY) * 1.0e6
    dK_i = -(I_K1 + I_to + I_Kr + I_Ks + I_pK - 2.0 * I_NaK) *
           CM_CELL / (V_C * FARADAY) * 1.0e6

    ds_all = SVector(
        dXr1, dXr2, dXs, dm, dh, dj, dd, df, df2, dfCass,
        ds, dr, dCa_i, dCa_SR, dCa_ss, dRp, dNa_i, dK_i,
    )
    return (-I_ion + stim, ds_all)
end

#--------------------------------------------------------------------------------# CytoZoo interface

"""
    TenTusscherEpi(; stim = Stimulus(; amplitude = 0))

ten Tusscher–Panfilov 2006 epicardial kinetics as a `CytoZoo.AbstractCardiacCellModel`:
19 states, `V` first.

`stim` defaults to zero amplitude — in tissue the stimulus belongs to the framework's
diffusion half, and a nonzero default would silently add a second one. The sign
convention is CytoZoo's: `Istim` is *subtracted* from `dV/dt`, so a **negative**
amplitude depolarizes. Currents here are in pA/pF ≡ mV/ms, so no capacitance division
is involved.

!!! note "Not a pre-paced steady state"
    `default_initial_state` is Niederer 2011 Table 2 — the benchmark's specified initial
    condition, which is what makes its activation times reproducible. It is not the
    limit cycle of a paced cell.
"""
struct TenTusscherEpi{S} <: CytoZoo.AbstractCardiacCellModel
    stim::S
end

TenTusscherEpi(; stim = Stimulus(; amplitude = 0.0)) = TenTusscherEpi(stim)

const STATE_NAMES = (
    :V, :Xr1, :Xr2, :Xs, :m, :h, :j, :d, :f, :f2, :fCass,
    :s, :r, :Ca_i, :Ca_SR, :Ca_ss, :Rprime, :Na_i, :K_i,
)

CytoZoo.num_states(::TenTusscherEpi) = NSTATES + 1
CytoZoo.transmembrane_potential_index(::TenTusscherEpi) = 1
CytoZoo.state_names(::TenTusscherEpi) = STATE_NAMES
CytoZoo.state_index(::TenTusscherEpi, n::Symbol) = findfirst(==(n), STATE_NAMES)

function CytoZoo.default_initial_state(::TenTusscherEpi)
    u = Vector{Float64}(undef, NSTATES + 1)
    u[1] = initial_voltage()
    s = initial_state()
    @inbounds for k in 1:NSTATES
        u[k + 1] = s[k]
    end
    return u
end

# `u` arrives as a strided view into FlashBang's state-blocked solution vector, so the
# gating states are gathered into an `SVector` before `rhs` sees them — that is what lets
# the kinetics stay a pure function over a stack value, exactly as in the MFO port.
@inline _gather_states(u) = SVector{NSTATES,Float64}(ntuple(k -> u[k + 1], Val(NSTATES)))

function _tt06_rhs!(du, u, model::TenTusscherEpi, x, t)
    # `rhs` takes the stimulus as a positive-depolarizing contribution to dV/dt, while
    # CytoZoo's convention is a negative-inward current subtracted from it. One sign flip
    # reconciles the two.
    stim = -model.stim(x, t)
    dV, ds = rhs(u[1], _gather_states(u), stim)
    du[1] = dV
    @inbounds for k in 1:NSTATES
        du[k + 1] = ds[k]
    end
    return nothing
end

function (model::TenTusscherEpi)(du, u, ::Nothing, t)
    _tt06_rhs!(du, u, model, nothing, t)
    return nothing
end

function (model::TenTusscherEpi)(du, u, p::SpatialContext, t)
    _tt06_rhs!(du, u, model, p.x, t)
    return nothing
end

end # module

using .TenTusscher2006

#--------------------------------------------------------------------------------# Single-cell verification

# These sit outside the `if` below on purpose: a `using` inside the same top-level
# expression takes effect too late for the `@printf` calls in that expression to expand.
using CytoZoo: default_initial_state, num_states, state_names, Stimulus
using Printf

# Only run as a script — the benchmark includes this file for the kinetics.
if abspath(PROGRAM_FILE) == @__FILE__
    const BCL = 1000.0        # pacing cycle length (ms)
    const NBEATS = 10
    const STIM_AMP = -52.0    # negative-inward, i.e. +52 mV/ms of depolarization
    const STIM_DUR = 1.0      # ms

    """
    Pace a single cell for `NBEATS` beats at `BCL` with forward Euler, and return the last
    beat's `(t, V)` trace plus the calcium diagnostics. Ten beats is enough for the
    concentrations to settle onto their limit cycle from the Table 2 initial conditions.

    This drives the model through the CytoZoo functor rather than through `rhs` directly,
    so it checks the wrapper — the state gather, the layout, and the stimulus sign — and
    not only the kinetics.
    """
    function pace(dt)
        model = TenTusscherEpi(;
            stim = Stimulus(; amplitude = STIM_AMP, period = BCL, duration = STIM_DUR),
        )
        u = default_initial_state(model)
        du = similar(u)
        nsub = round(Int, BCL / dt)
        ts, Vs = Float64[], Float64[]
        cai_peak = 0.0
        for beat in 1:NBEATS, k in 0:(nsub - 1)
            tb = k * dt                             # time within this beat
            if beat == NBEATS
                push!(ts, tb)
                push!(Vs, u[1])
                cai_peak = max(cai_peak, u[14])     # Ca_i
            end
            model(du, u, nothing, tb)
            @. u += dt * du
        end
        return (; ts, Vs, cai_peak, ca_sr = u[15])
    end

    "Resting/peak potential, maximum upstroke velocity, and APD90 of one beat."
    function ap_metrics(ts, Vs, dt)
        V_rest = Vs[1]
        V_peak, ipeak = findmax(Vs)
        dVdt_max = maximum(diff(Vs)) / dt
        iact = findfirst(>(0.0), Vs)
        V90 = V_rest + 0.1 * (V_peak - V_rest)
        irep = findfirst(i -> Vs[i] < V90, ipeak:length(Vs))
        apd90 = (iact === nothing || irep === nothing) ? NaN :
                ts[ipeak + irep - 1] - ts[iact]
        return (; V_rest, V_peak, dVdt_max, apd90)
    end

    model = TenTusscherEpi()
    println("ten Tusscher-Panfilov 2006 epicardial - single-cell verification")
    @printf "  %d states, V first: %s ...\n" num_states(model) join(state_names(model)[1:5], ", ")
    @printf "  %d beats at BCL = %.0f ms, stimulus %.0f (negative-inward) for %.0f ms\n\n" NBEATS BCL STIM_AMP STIM_DUR

    results = Dict{Float64,Any}()
    for dt in (0.01, 0.002)
        r = pace(dt)
        m = ap_metrics(r.ts, r.Vs, dt)
        results[dt] = (; m..., cai_peak_uM = 1.0e3 * r.cai_peak, r.ca_sr)
        @printf "  dt = %.3f ms:  V_rest %7.2f   V_peak %6.2f   dV/dt_max %6.1f   APD90 %5.1f   Ca_i peak %.3f uM   Ca_SR %.2f mM\n" dt m.V_rest m.V_peak m.dVdt_max m.apd90 (1.0e3 * r.cai_peak) r.ca_sr
    end

    # Single-cell targets, same bands MatrixFreeOperators' port checks against. This is
    # the port's acceptance test: the tissue benchmark is meaningless if the kinetics
    # are wrong, and a wrong sign or a mis-gathered state shows up here first.
    ref = results[0.01]
    checks = (
        ("V_rest ∈ [-87, -85] mV", -87.0 <= ref.V_rest <= -85.0),
        ("V_peak ∈ [35, 50] mV", 35.0 <= ref.V_peak <= 50.0),
        ("dV/dt_max ∈ [340, 430] mV/ms (isolated cell)", 340.0 <= ref.dVdt_max <= 430.0),
        ("APD90 ∈ [270, 310] ms", 270.0 <= ref.apd90 <= 310.0),
        ("Ca_i transient peak ∈ [0.6, 1.4] µM", 0.6 <= ref.cai_peak_uM <= 1.4),
        ("Ca_SR load ∈ [3.0, 4.0] mM (no drift)", 3.0 <= ref.ca_sr <= 4.0),
    )
    println()
    for (label, ok) in checks
        @printf "  [%s] %s\n" (ok ? "pass" : "FAIL") label
    end
    all(last, checks) || error("single-cell verification failed")
end
