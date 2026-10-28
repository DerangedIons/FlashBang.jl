#--------------------------------------------------------------------------------# Adaptive refinement

"""
    regrid(split, disc, f, u; refine, coarsen = Returns(false)) -> (f′, u′)

Adapt the `BlockForest` that `f` was semidiscretized on to the current solution `u`, and
return the problem on the new leaf set: a fresh split function `f′` and the solution `u′`
carried across, in the same state-blocked layout.

The criteria see each leaf of the voltage gradient `∇φₘ` as an ordinary MatrixFreeOperators
`Field` — one `SVector` per cell, with the leaf grid attached — so a criterion is a
reduction such as `b -> maximum(norm, interior(b)) > τ`, and `b.grid.extent` is there for a
geometric rule. Refine marks at `maxlevel` and coarsen marks on level 0 are ignored, and 2:1
balance is restored, as in MatrixFreeOperators' `regrid!`.

All cell states move together, as one `SVector` per cell, with the slope-limited
conservative transfer: a refined cell gets its parent's value plus a minmod-limited slope,
so no state leaves the range of its neighbours (gates stay in [0, 1], concentrations stay
positive), and a coarsened cell gets the mean of its children, so each state's volume
integral is unchanged.

An integrator cannot change size, so the caller rebuilds it on `(f′, u′)` at the current
time. When the marks change nothing, `(f, u)` are returned as they are (`f′ === f`), and
the existing integrator stays valid.

CPU only: the forest topology is edited on the host.

### Examples

```julia
using LinearAlgebra: norm

(f, u) = regrid(split, disc, f, integrator.u;
                refine  = b -> maximum(norm, interior(b)) > 60,
                coarsen = b -> maximum(norm, interior(b)) < 20)
f === old_f || (integrator = init(OperatorSplittingProblem(f, u, (t, tend)), alg; dt))
```
"""
function regrid(
    split::ReactionDiffusionSplit{<:MonodomainModel},
    disc::FiniteDifferenceDiscretization,
    f::GenericSplitFunction,
    u::AbstractVector;
    refine,
    coarsen=Returns(false),
)
    forest = _diffusion_function(f).grid
    forest isa BlockForest || throw(
        ArgumentError(
            "regrid needs a problem semidiscretized on a BlockForest; this one is on a " *
            "$(nameof(typeof(forest)))",
        ),
    )
    N = num_nodes(f)
    M = _num_states(f)
    length(u) == N * M || throw(
        DimensionMismatch("u has $(length(u)) entries, the problem $(N * M) ($N nodes × $M states)")
    )

    k = CytoZoo.transmembrane_potential_index(split.model.ion)
    φ = scalar_field(forest, eltype(u))
    flat_to_interior!(φ, view(u, _state_block(k, N)))
    ∇φ = gradient(forest) * φ
    states = _state_field(u, forest, N, Val(M))

    (∇φ′, states′) = regrid!(∇φ, states; refine, coarsen)
    ∇φ′ === ∇φ && return (f, u)

    f′ = semidiscretize(split, disc, forest)
    return (f′, _state_vector(states′, num_nodes(f′), M))
end

# The state-blocked vector as one `SVector{M}` per cell, so a single `regrid!` moves every
# state through the same topology change. Node `n` of `u` is cell `n` of the forest's flat
# order: leaf by leaf, column-major within each leaf.
function _state_field(u, forest::BlockForest, N, ::Val{M}) where {M}
    T = eltype(u)
    padded = forest.blocksize .+ 2 .* forest.halo
    blocks = [zeros(SVector{M,T}, padded...) for _ in 1:nleaves(forest)]
    states = BlockField{Center,SlopeLimited}(blocks, forest)
    n = 0
    for i in 1:nleaves(forest)
        cells = interior(block(states, i))
        for I in CartesianIndices(cells)
            n += 1
            cells[I] = SVector{M,T}(ntuple(m -> u[(m - 1) * N + n], Val(M)))
        end
    end
    return states
end

function _state_vector(states, N, M)
    u = Vector{eltype(eltype(states))}(undef, N * M)
    n = 0
    for i in 1:nleaves(states.grid)
        cells = interior(block(states, i))
        for I in CartesianIndices(cells)
            n += 1
            s = cells[I]
            for m in 1:M
                u[(m - 1) * N + n] = s[m]
            end
        end
    end
    n == N || error("forest has $n cells but the new problem $N nodes")
    return u
end
