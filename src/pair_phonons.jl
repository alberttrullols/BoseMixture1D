using Roots: find_zero, Brent
using Printf

# Luttinger parameter K and its inverse alpha = 1/K for the 1D repulsive Lieb-Liniger gas,
# from the scattering length a (gamma = |2/a| = g at unit density, hbar = m = 1). The two
# branches are standard weak/strong-coupling approximations for K(gamma) (Cazalilla 2004).
function kpar_from_a(a::Float64)
    gamma = abs(2.0 / a)
    if gamma > 8.0
        Kpar = 1.0 + 4.0 / gamma
    else
        Kpar = π / sqrt(gamma) / sqrt(1.0 - sqrt(gamma) / (2.0 * π))
    end
    return Kpar
end

function alpha_from_a(a::Float64)
    return 1.0 / kpar_from_a(a)
end

function print_kpar_alpha(a::Float64)
    Kpar = kpar_from_a(a)
    alpha = alpha_from_a(a)
    @printf("a = %.12f\n", a)
    @printf("gamma = %.12f\n", abs(2.0 / a))
    @printf("Kpar = %.12f\n", Kpar)
    @printf("alpha = 1/Kpar = %.12f\n", alpha)
end

# Pure two-body Bethe-Peierls resonance cos(k*r+delta) spanning the whole box (0, L/2), i.e.
# the PairCosine solution; used as a fallback when no interior matching point exists.
@inline function edge_bethe_peierls_constants(a::Float64, L::Float64)
    f(delta) = (2.0 * (-delta) / L) * tan(delta) - 1.0 / a
    tol = 1e-12
    delta = find_zero(f, (tol, π / 2 - tol), Brent())
    k = -2.0 * delta / L
    return k, L / 2.0, 1.0, -delta / k
end

@inline @fastmath function Rpar_equation(R::Float64, alpha::Float64, a::Float64, L::Float64)
    if R <= 0.0 || R >= L / 2.0
        return NaN
    end
    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * R / L) * tan(π * R / L)) - 1.0)
    if disc <= 0.0
        return NaN
    end
    k = sqrt(disc)
    return -k * tan(k * R + atan(1.0 / (k * a))) - alpha * (π / L) / tan(π * R / L)
end

# Bisection solve for the matching radius Rpar where the short-range piece meets the
# long-distance Luttinger-liquid tail with continuous value and slope. Falls back to L/2
# (handled by the caller) when no interior root exists.
@inline @fastmath function solve_Rpar(alpha::Float64, a::Float64, L::Float64)
    if alpha <= 0.0 || a == 0.0
        return L / 2.0
    end

    precision = 1e-12
    Rmin = 1e-8
    Rmax = L / 2.0 - 1e-8

    y_min = Rpar_equation(Rmin, alpha, a, L)
    y_max = Rpar_equation(Rmax, alpha, a, L)

    if isfinite(y_max) && abs(y_max) < precision
        return Rmax
    end
    if isfinite(y_min) && isfinite(y_max) && y_min * y_max > 0.0
        return L / 2.0
    end

    iter = 1
    while true
        R = (Rmin + Rmax) / 2.0
        y = Rpar_equation(R, alpha, a, L)

        if isfinite(y) && isfinite(y_min) && y * y_min < 0.0
            Rmax = R
            y_max = y
        else
            Rmin = R
            y_min = y
        end

        if abs(y) < precision
            return R
        end

        if iter >= 1000
            return (Rmax * y_min - Rmin * y_max) / (y_min - y_max)
        end
        iter += 1
    end
end

# Short-distance constants (k, Rpar, Atrial, Btrial) for the piece
# log|psi(r)| = log|Atrial * cos(k * (r - Btrial))|  for r < Rpar,
# continuously matched (value and slope) to the long-distance Luttinger tail
# alpha * log(sin(pi r / L)) at r = Rpar. Falls back to the full-box Bethe-Peierls
# solution (Rpar = L/2, alpha-independent) when no interior matching point exists.
@inline @fastmath function short_range_analytic_constants(alpha::Float64, a::Float64, L::Float64)
    if alpha <= 0.0 || a == 0.0
        return 0.0, L / 2.0, 1.0, 0.0
    end
    a < 0.0 || throw(ArgumentError("short_range_analytic_constants needs repulsive a < 0, got $a"))

    Rpar = solve_Rpar(alpha, a, L)
    if !(Rpar < L / 2.0)
        return edge_bethe_peierls_constants(a, L)
    end

    disc = -alpha * (π / L)^2 * ((alpha - 1.0) / (tan(π * Rpar / L) * tan(π * Rpar / L)) - 1.0)
    if disc <= 0.0
        return edge_bethe_peierls_constants(a, L)
    end
    k = sqrt(disc)
    if !isfinite(k)
        return edge_bethe_peierls_constants(a, L)
    end

    Btrial = -atan(1.0 / (k * a)) / k
    Atrial = (sin(π * Rpar / L))^alpha / cos(k * (Rpar - Btrial))
    if !isfinite(Btrial) || !isfinite(Atrial)
        return edge_bethe_peierls_constants(a, L)
    end

    return k, Rpar, Atrial, Btrial
end

# Called once per run (not per step): short_range_analytic_constants falls back to
# alpha-independent edge constants when no matching point exists, which it reports as Rpar = L/2.
function warn_missing_matching_point(label::String, alpha::Float64, a::Float64, L::Float64)
    (alpha <= 0.0 || a == 0.0) && return nothing
    _, Rpar, _, _ = short_range_analytic_constants(alpha, a, L)
    if Rpar >= L / 2.0
        println("WARNING: No matching point found for $label (alpha = $alpha, K = $(1.0 / alpha), a = $a, L = $L); ",
                "falling back to L/2 edge constants, which ignore alpha.")
    end
    return nothing
end

# Short-range matched piece glued to the long-distance Luttinger-liquid phonon tail
# alpha * log(sin(pi r / L)) (alpha = 1/K, K the Luttinger parameter) at a self-consistently
# solved matching radius Rpar (2b_phonons): continuous value and slope at r = Rpar, recovering
# the bare two-body solution (PairCosine) when no interior matching point exists.
struct PairPhonon <: PairTrial
    k::NTuple{3,Float64}
    Rpar::NTuple{3,Float64}
    Atrial::NTuple{3,Float64}
    Btrial::NTuple{3,Float64}
    alpha::NTuple{3,Float64}
    L::Float64
end

function PairPhonon(a::NTuple{3,Float64}, alpha::NTuple{3,Float64}, L::Float64)
    all(ai -> ai < 0.0, a) || throw(ArgumentError("PairPhonon needs repulsive a < 0, got $a"))
    all(>=(0.0), alpha) || throw(ArgumentError("PairPhonon needs alpha >= 0 (alpha = 1/K), got $alpha"))
    constants = map((ai, alphai) -> short_range_analytic_constants(alphai, ai, L), a, alpha)
    return PairPhonon(map(c -> c[1], constants), map(c -> c[2], constants),
                       map(c -> c[3], constants), map(c -> c[4], constants), alpha, L)
end

@inline function pair_eval(tab::PairPhonon, r::Float64, s::Int)
    alpha = tab.alpha[s]
    alpha == 0.0 && return 0.0, 0.0, 0.0

    k = tab.k[s]
    Rpar = tab.Rpar[s]
    if isfinite(k) && r < Rpar
        θ = k * (r - tab.Btrial[s])
        c = cos(θ)
        u = log(abs(tab.Atrial[s] * c))
        u1 = -k * tan(θ)
        u2 = -k * k / (c * c)
        return u, u1, u2
    end

    L = tab.L
    θ = π * r / L
    sn = sin(θ)
    u = alpha * log(sn)
    u1 = alpha * (π / L) * (cos(θ) / sn)
    u2 = -alpha * (π / L)^2 / (sn * sn)
    return u, u1, u2
end
