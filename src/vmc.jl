if !isdefined(@__MODULE__, :log_psiT)
    include("core.jl")
end
if !isdefined(@__MODULE__, :run_reblocking_loop)
    include("reblocking.jl")
end
using Printf

# MC parameters
const VMC_EQUILIBRATION_STEPS = 20_000
const VMC_PRODUCTION_STEPS = 50_000
const VMC_STEP_SIZE = 4.0

"""
Standard Metropolis VMC for a two-component Bose gas with fixed N1 and N2 particles.
Species labels are assigned once at initialization and never change (no tunneling between
species), so only particle positions are updated.
"""
function run_metropolis(tab::PairTrial;
                       L :: Float64=0.0, # by default set from N_total and n if not provided
                       N1::Int, N2::Int, seed::Int=1234, verbose::Bool=true,
                       equilibration_steps::Int=VMC_EQUILIBRATION_STEPS,
                       production_steps::Int=VMC_PRODUCTION_STEPS,
                       step_size::Float64=VMC_STEP_SIZE)
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    production_steps > 0 || throw(ArgumentError("production_steps must be positive"))
    step_size > 0 || throw(ArgumentError("step_size must be positive"))
    total_steps = equilibration_steps + production_steps
    N_total = N1 + N2
    n = 1.0
    if L == 0.0
        L = N_total / n
    end
    invL = 1.0 / L

    rng = MersenneTwister(seed)

    # Initialize positions and species labels (Float64)
    x = rand(rng, N_total) .* L
    spin = zeros(Float64, N_total)
    for i in 1:N1
        spin[i] = 1.0
    end
    for i in N1+1:N_total
        spin[i] = 2.0
    end

    # Pre-allocate buffers
    F = zeros(N_total)

    logpsi = log_psiT(x, spin, tab; N_total, L, invL)

    E_lap_acc = 0.0
    E_lap_acc2 = 0.0
    nmeas = 0

    Nbins_corr = 100
    dr = (L/2) / Nbins_corr
    k_min = 2π / L
    Nk_vals = 100
    Cs      = zeros(Float64, Nbins_corr)
    Cs_count = zeros(Int, Nbins_corr)
    Cn      = zeros(Float64, Nbins_corr)
    Sk_acc  = zeros(Float64, Nk_vals)
    Sn_acc  = zeros(Float64, Nk_vals)
    Sk_count = 0
    n_pos_accepted = 0
    n_pos_attempted = 0

    for step in 1:total_steps
        if step % 10000 == 0
            println("Step ", step, "/", total_steps)
            @printf("  Position acceptance rate so far: %.4f\n",
                    n_pos_attempted > 0 ? n_pos_accepted / n_pos_attempted : 0.0)
            flush(stdout)
        end

        # Pick random particle and attempt move
        i = rand(rng, 1:N_total)
        x_new = copy(x)
        x_new[i] = pbc(x_new[i] + step_size * (rand(rng) - 0.5), L, invL)

        logpsi_new = log_psiT(x_new, spin, tab; N_total, L, invL)
        acc = min(1.0, exp(2.0 * (logpsi_new - logpsi)))
        n_pos_attempted += 1

        if rand(rng) < acc
            x = x_new
            logpsi = logpsi_new
            n_pos_accepted += 1
        end

        # Measurements after equilibration
        if step > equilibration_steps
            E_lap = turbo_compute_drift_and_energy!(F, x, spin, tab; N_total, L, invL)
            E_lap_acc += E_lap
            E_lap_acc2 += E_lap^2
            nmeas += 1

            # Cs(r) and Cn(r)
            for i in 1:N_total-1
                si = (spin[i] == 1.0) ? 1.0 : -1.0
                xi = x[i]
                for j in i+1:N_total
                    sj = (spin[j] == 1.0) ? 1.0 : -1.0
                    xj = x[j]
                    dx = xi - xj
                    dx -= L * round(dx * invL)
                    r = abs(dx)
                    if r < L/2
                        bin = Int(floor(r / dr)) + 1
                        if bin <= Nbins_corr
                            Cs[bin]       += si * sj
                            Cs_count[bin] += 1
                            Cn[bin]       += 1.0
                        end
                    end
                end
            end

            # Sk(k) and Sn(k)
            for k_idx in 1:Nk_vals
                k_val = (k_idx - 1) * k_min
                re_spin = 0.0; im_spin = 0.0
                re_dens = 0.0; im_dens = 0.0
                for i in 1:N_total
                    s  = (spin[i] == 1.0) ? 1.0 : -1.0
                    xi = x[i]
                    re_spin += s * cos(k_val * xi)
                    im_spin += s * sin(k_val * xi)
                    re_dens += cos(k_val * xi)
                    im_dens += sin(k_val * xi)
                end
                Sk_acc[k_idx] += (re_spin^2 + im_spin^2) / N_total
                Sn_acc[k_idx] += (re_dens^2 + im_dens^2) / N_total
            end
            Sk_count += 1
        end
    end

    E_lap_mean = E_lap_acc / nmeas
    E_lap_var = E_lap_acc2 / nmeas - E_lap_mean^2
    E_lap_var = max(0.0, E_lap_var)
    E_lap_err = sqrt(E_lap_var / nmeas)

    pos_acc_rate = n_pos_attempted > 0 ? n_pos_accepted / n_pos_attempted : 0.0
    println("Final position acceptance rate: ", round(pos_acc_rate, digits=4),
            " (", n_pos_accepted, "/", n_pos_attempted, ")")

    mkpath("VMC")
    open("VMC/results.txt", "w") do f
        @printf(f, "N1                  %d\n", N1)
        @printf(f, "N2                  %d\n", N2)
        @printf(f, "E_VMC               %.15f\n", E_lap_mean)
        @printf(f, "E_VMC_err           %.6e\n", E_lap_err)
        @printf(f, "E_VMC_per_particle  %.15f\n", E_lap_mean / N_total)
        @printf(f, "pos_acceptance_rate %.6f\n", pos_acc_rate)
        @printf(f, "equilibration_steps %d\n", equilibration_steps)
        @printf(f, "production_steps    %d\n", production_steps)
    end

    # Normalize and save correlators / structure factors
    Cs_full = zeros(Float64, Nbins_corr)
    for i in 1:Nbins_corr
        if Cs_count[i] > 0
            Cs_full[i] = Cs[i] / Cs_count[i]
        end
    end
    expected_total_pairs = nmeas * (N_total * (N_total - 1) / 2.0) * (dr / (L/2))
    Cn_full = Cn ./ expected_total_pairs
    Sk_avg  = Sk_count > 0 ? Sk_acc ./ Sk_count : zeros(Float64, Nk_vals)
    Sn_avg  = Sk_count > 0 ? Sn_acc ./ Sk_count : zeros(Float64, Nk_vals)

    base = @sprintf("N1%d_N2%d_vmc", N1, N2)
    N_folder = "N$N_total"
    mkpath("VMC/correlators/$N_folder")
    mkpath("VMC/spin_structure_factor/$N_folder")
    mkpath("VMC/density_structure_factor/$N_folder")
    open("VMC/correlators/$N_folder/Cs_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cs_full[i])
        end
    end
    open("VMC/correlators/$N_folder/Cn_full_$base.txt", "w") do f
        for i in 1:Nbins_corr
            r = (i - 0.5) * dr
            println(f, r, " ", Cn_full[i])
        end
    end
    open("VMC/spin_structure_factor/$N_folder/Sk_$base.txt", "w") do f
        println(f, "# k  S_s(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sk_avg[k_idx])
        end
    end
    open("VMC/density_structure_factor/$N_folder/Sn_$base.txt", "w") do f
        println(f, "# k  S_n(k)")
        for k_idx in 1:Nk_vals
            k_val = (k_idx - 1) * k_min
            println(f, k_val, " ", Sn_avg[k_idx])
        end
    end

    if verbose
        println("VMC result (two-component Bose gas, fixed species labels):")
        println("Energy (via Laplacian) = ", E_lap_mean/L, " ± ", E_lap_err/L)
        println("Measurements = ", nmeas)
    end

    return E_lap_mean, E_lap_err
end
