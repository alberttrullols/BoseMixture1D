using Random

include("pair_analytic.jl")
include("pair_phonons.jl")

const hbar = 1.0
const m = 1.0
const D_const = hbar^2 / (2.0 * m)

# periodic boundary conditions
@inline @fastmath pbc(dx, L, invL) = dx - L * round(dx * invL)

# Pair sector from species labels (1.0 = species 1, 2.0 = species 2): 1 = 11, 2 = 12, 3 = 22
@inline pair_sector(si::Float64, sj::Float64) = si == sj ? (si == 1.0 ? 1 : 3) : 2

# log of trial wavefunction: sum over pairs of the pair log-factor. Species labels are fixed
# per particle (no inter-species tunneling), so this only depends on positions and the
# (constant) species assignment.
function log_psiT(x::AbstractVector{Float64}, spin::AbstractVector{Float64},
                  tab::PairTrial;
                  N_total::Int, L::Float64, invL::Float64)
    s = 0.0
    @inbounds for i in 1:N_total-1
        xi = x[i]
        si = spin[i]
        for j in i+1:N_total
            dx = abs(pbc(xi - x[j], L, invL))
            u, _, _ = pair_eval(tab, dx, pair_sector(si, spin[j]))
            s += u
        end
    end
    return s
end

# Drift force and kinetic energy in a single vectorized pass.
function turbo_compute_drift_and_energy!(
    F::AbstractVector{Float64},
    x::AbstractVector{Float64},
    spin::AbstractVector{Float64},
    tab::PairTrial;
    N_total::Int, L::Float64, invL::Float64
)::Float64

    fill!(F, 0.0)
    lap = 0.0

    @inbounds for i in 1:N_total
        xi = x[i]
        si = spin[i]
        fi = 0.0
        lap_i = 0.0
        for j in 1:N_total
            j == i && continue
            dx = xi - x[j]
            dx = dx - L * floor(dx * invL + 0.5)
            absdx = abs(dx)
            sgn = dx > 0.0 ? 1.0 : -1.0
            sj = spin[j]

            u, u1, u2 = pair_eval(tab, absdx, pair_sector(si, sj))
            fi += u1 * sgn
            lap_i -= u2
        end
        F[i] = fi
        lap += lap_i
    end

    # F^2 sum for kinetic energy
    F2_sum = 0.0
    @inbounds @simd for i in 1:N_total
        F2_sum += F[i] * F[i]
    end

    # Full N×N loop double-counts pairs
    kinetic_part = (hbar^2 / (2.0 * m)) * (lap - F2_sum)
    return kinetic_part
end
