using Roots: find_zero, Brent

# Pair trial interface: pair_eval(tab, r, s) -> (u, u', u'') of log|psi_s(r)| for sector s in 1:3 (11, 12, 22).
abstract type PairTrial end

# Analytic per-sector pair trials for the two-body problem of two non-tunneling species.
# Sector order is (11, 12, 22), matching pair_sector in core.jl. Scattering lengths a = -2/g (hbar = m = 1).

# Bare cosine: u = log cos(k r + delta), with k = -2 delta / L (2b_uncoupled).
struct PairCosine <: PairTrial
    k::NTuple{3,Float64}
    delta::NTuple{3,Float64}
end

function compute_k_delta(a::Float64, L::Float64)
    tol = 1e-20
    f(delta) = (2.0 * (-delta) / L) * tan(delta) - 1.0 / a
    delta = find_zero(f, (tol, π / 2 - tol), Brent())
    k = -2.0 * delta / L
    return k, delta
end

function PairCosine(a::NTuple{3,Float64}, L::Float64)
    all(x -> x < 0.0, a) || throw(ArgumentError("PairCosine needs repulsive a < 0, got $a"))
    kd = map(ai -> compute_k_delta(ai, L), a)
    return PairCosine(map(first, kd), map(last, kd))
end

@inline function pair_eval(tab::PairCosine, r::Float64, s::Int)
    k = tab.k[s]
    θ = k * r + tab.delta[s]
    c = cos(θ)
    u = log(c)
    u1 = -k * tan(θ)
    u2 = -k * k / (c * c)
    return u, u1, u2
end

# Uncoupled trial with a matching point R per sector (2b_uncoupled_matching): the Bethe-Peierls
# cosine cos(k (r - R)) on [0, R], constant beyond R. For R = L/2 it reduces to PairCosine.
struct PairMatched <: PairTrial
    R::NTuple{3,Float64}
    k::NTuple{3,Float64}
end

# Zero slope at R plus Bethe-Peierls at r = 0 give x tan(x) = -R/a with x = kR in (0, π/2).
function pair_matched(a::NTuple{3,Float64}, R::NTuple{3,Float64})
    all(ai -> ai < 0.0, a) || throw(ArgumentError("PairMatched needs repulsive a < 0, got $a"))
    all(Ri -> Ri > 0.0, R) || throw(ArgumentError("matching radii must be positive, got $R"))
    x = map((ai, Ri) -> find_zero(y -> y * tan(y) + Ri / ai, (1e-12, π / 2 - 1e-12), Brent()), a, R)
    return PairMatched(R, map((xi, Ri) -> xi / Ri, x, R))
end

@inline function pair_eval(tab::PairMatched, r::Float64, s::Int)
    r >= tab.R[s] && return 0.0, 0.0, 0.0
    k = tab.k[s]
    θ = k * (r - tab.R[s])
    c = cos(θ)
    return log(c), -k * tan(θ), -k * k / (c * c)
end
