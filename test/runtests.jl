using FlashBang
using Test
using JET

# Explicit imports, as in the package itself: MatrixFreeOperators exports `solve` and
# `gradient`, and `using` it wholesale would shadow the SciML stack.
using KernelAbstractions: KernelAbstractions
using MatrixFreeOperators: MatrixFreeOperators
using OrdinaryDiffEqOperatorSplitting: OrdinaryDiffEqOperatorSplitting
using CytoZoo: CytoZoo

using CytoZoo: FHNModel, SpatialStep, Stimulus
using OrdinaryDiffEqLowOrderRK: Euler

include("testutils.jl")

@testset "FlashBang.jl" begin
    @testset "Code linting (JET.jl)" begin
        # `target_modules` rather than the template's `target_defined_modules = true`: JET
        # 0.12 dropped that configuration name (it validates configs strictly and throws a
        # `JETConfigError`). This is the same restriction — report only problems inside
        # FlashBang's own module context, not inside its dependencies.
        JET.test_package(FlashBang; target_modules=(FlashBang,))
    end
    include("test_pipeline.jl")
    include("test_diffusion.jl")
    include("test_cable.jl")
    include("test_gpu.jl")
end
