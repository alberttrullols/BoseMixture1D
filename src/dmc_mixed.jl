if !isdefined(@__MODULE__, :turbo_compute_drift_and_energy!)
    include("core.jl")
end
if !isdefined(@__MODULE__, :run_reblocking_loop)
    include("reblocking.jl")
end

using Statistics, Printf
using Base.Threads

# Diffusion constant
if !@isdefined(D_const)
    const D_const = hbar^2 / (2.0 * m)
end

# Maximum number of copies per walker
if !@isdefined(MAX_COPIES)
    const MAX_COPIES = 1000
end

# Pre-allocated workspace
mutable struct DMCWorkspace_2comp
    walker_positions::Matrix{Float64}   # (num_walkers, N_total)
    walker_spin::Matrix{Float64}        # (num_walkers, N_total): 1.0 or 2.0, fixed species label
    new_positions::Matrix{Float64}      # buffer for next generation
    new_spin::Matrix{Float64}           # buffer for spin after branching
    M::Vector{Int}                      # copies per walker
    E_local::Vector{Float64}            # local energy per walker
    new_E_local::Vector{Float64}        # replicated local energies after branching
end

function DMCWorkspace_2comp(num_walkers::Int, N_total::Int)
    max_walkers = num_walkers * MAX_COPIES
    return DMCWorkspace_2comp(
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Vector{Int}(undef, num_walkers),
        Vector{Float64}(undef, num_walkers),
        Vector{Float64}(undef, max_walkers)
    )
end

"""
Run Diffusion Monte Carlo for 1D two-component bosons with contact interactions.
Species labels are fixed for the whole run (no inter-species tunneling).
Multi-threaded version, returns mixed estimators and writes flat-file output.
"""
function DMC_mixed(num_steps::Int, num_walkers::Int, dt::Float64, E_T_init::Float64,
    tab::PairTrial,
    N1::Int, N2::Int;
    seed::Int=1234,
    importance_sampling::Bool=true,
    branching::Bool=true,
    time_order::Int=1,
    verbose::Bool=true,
    reblocking::Bool=true,
    num_threads_to_use::Int=Threads.nthreads(),
    equilibration_steps::Int=0,
    observables::Set{Symbol}=Set{Symbol}(),
    save_state_path::Union{Nothing,String}=nothing,
)
    0 <= equilibration_steps < num_steps ||
        throw(ArgumentError("equilibration_steps must be between 0 and num_steps - 1"))
    N_total = N1 + N2
    n = 1.
    L = N_total / n
    invL = 1.0 / L

    #################################################################### parameter initialization
    t_start = time_ns()
    rng = MersenneTwister(seed)
    num_walkers_initial = num_walkers
    diffusion_scale = sqrt(2.0 * D_const * dt)
    E_T = E_T_init
    C_pop = 1.0 / dt
    energies = zeros(Float64, num_steps)
    energies_squared = zeros(Float64, num_steps)

    equilibration_cutoff = equilibration_steps
    k_min = 2π / L
    Nk_vals = 75
    Sk_acc = zeros(Float64, Nk_vals)
    Sk_count = 0
    Sn_acc = zeros(Float64, Nk_vals)
    Sn_count = 0

    # Spin correlation Cs(r): binned average of s_i s_j over pairs at minimum-image distance r
    Nbins_corr = 100
    dr = (L / 2) / Nbins_corr
    Cs_bins = zeros(Float64, Nbins_corr)
    Cs_bin_count = zeros(Int, Nbins_corr)

    ws = DMCWorkspace_2comp(num_walkers, N_total)

    #################################################################### walkers initialization
    @inbounds for w in 1:num_walkers
        for i in 1:N1
            ws.walker_positions[w, i] = rand(rng) * L
            ws.walker_spin[w, i] = 1.0
        end
        for i in 1:N2
            ws.walker_positions[w, N1 + i] = rand(rng) * L
            ws.walker_spin[w, N1 + i] = 2.0
        end
    end

    #################################################################### thread-local buffers
    num_threads_to_use = min(num_threads_to_use, Threads.nthreads())
    n_buffers = Threads.nthreads() + 1  # +1 to cover potential interactive thread
    x_buffers             = [zeros(N_total) for _ in 1:n_buffers]
    drift_buffers         = [zeros(N_total) for _ in 1:n_buffers]
    drift_old_buffers     = [zeros(N_total) for _ in 1:n_buffers]
    rand_buffers          = [zeros(N_total) for _ in 1:n_buffers]
    spin_buffers          = [zeros(Float64, N_total) for _ in 1:n_buffers]
    rng_threads           = [MersenneTwister(seed + tid - 1) for tid in 1:n_buffers]

    #################################################################### dmc loop
    for step in 1:num_steps
        if length(ws.M) < num_walkers
            resize!(ws.M, num_walkers)
            resize!(ws.E_local, num_walkers)
        end

        # walker propagation
        Threads.@threads for w in 1:num_walkers
            tid = threadid()

            xb       = x_buffers[tid]
            F_old    = drift_old_buffers[tid]
            F_new    = drift_buffers[tid]
            randb    = rand_buffers[tid]
            spin_buf = spin_buffers[tid]
            rng_tid  = rng_threads[tid]

            # Load walker and spin
            @inbounds for i in 1:N_total
                xb[i]       = ws.walker_positions[w, i]
                spin_buf[i] = ws.walker_spin[w, i]
            end

            # Compute drift and energy at old position
            E_old = turbo_compute_drift_and_energy!(F_old, xb, spin_buf, tab; N_total, L, invL)

            # Propagate walker
            randn!(rng_tid, randb)

            if importance_sampling
                if time_order == 1
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(
                            xb[i] +
                            2.0 * D_const * F_old[i] * dt +
                            randb[i] * diffusion_scale,
                            L
                        )
                    end

                elseif time_order == 2
                    # Save F_old for predictor-corrector
                    @inbounds @simd for i in 1:N_total
                        F_new[i] = F_old[i]
                    end

                    # Predictor
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(xb[i] +
                                    2.0 * D_const * F_new[i] * dt +
                                    randb[i] * diffusion_scale, L)
                    end

                    # Compute drift at predictor position
                    turbo_compute_drift_and_energy!(F_new, xb, spin_buf, tab; N_total, L, invL)

                    # Corrector
                    @inbounds @simd for i in 1:N_total
                        xb[i] = mod(ws.walker_positions[w, i] +
                                    D_const * (F_old[i] + F_new[i]) * dt +
                                    randb[i] * diffusion_scale, L)
                    end
                end
            else
                # Pure diffusion, no drift
                @inbounds @simd for i in 1:N_total
                    xb[i] = mod(xb[i] + randb[i] * diffusion_scale, L)
                end
            end

            # Compute drift and energy at new position
            E_new = turbo_compute_drift_and_energy!(F_new, xb, spin_buf, tab; N_total, L, invL)

            ws.E_local[w] = E_new

            # Branching weight
            if branching
                if isfinite(E_old) && isfinite(E_new)
                    if time_order == 2
                        wi = exp(-dt * (E_old + E_new - 2.0 * E_T) / 2.0)
                    elseif time_order == 1
                        wi = exp(-dt * (E_old - E_T))
                    end
                    wi = min(wi, Float64(MAX_COPIES))
                    ws.M[w] = min(floor(Int, wi + rand(rng_tid)), MAX_COPIES)
                    if ws.M[w] == MAX_COPIES
                        println("Warning: MAX_COPIES reached for walker $w at step $step")
                    end
                else
                    ws.M[w] = 0
                end
            else
                ws.M[w] = 1
            end

            # Store walker and spin
            @inbounds for i in 1:N_total
                ws.walker_positions[w, i] = xb[i]
                ws.walker_spin[w, i]      = spin_buf[i]
            end
        end

        # Branching
        total_new = sum(ws.M[1:num_walkers])
        total_new == 0 && error("All walkers died.")

        if size(ws.new_positions, 1) < total_new
            ws.new_positions = Matrix{Float64}(undef, total_new, N_total)
        end
        if size(ws.new_spin, 1) < total_new
            ws.new_spin = Matrix{Float64}(undef, total_new, N_total)
        end
        if length(ws.new_E_local) < total_new
            resize!(ws.new_E_local, total_new)
        end

        idx = 1
        @inbounds for w in 1:num_walkers
            for _ in 1:ws.M[w]
                for i in 1:N_total
                    ws.new_positions[idx, i] = ws.walker_positions[w, i]
                    ws.new_spin[idx, i]      = ws.walker_spin[w, i]
                end
                ws.new_E_local[idx] = ws.E_local[w]
                idx += 1
            end
        end

        num_walkers = total_new

        if size(ws.walker_positions, 1) < num_walkers
            ws.walker_positions = Matrix{Float64}(undef, num_walkers, N_total)
        end
        if size(ws.walker_spin, 1) < num_walkers
            ws.walker_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        @inbounds for w in 1:num_walkers
            for i in 1:N_total
                ws.walker_positions[w, i] = ws.new_positions[w, i]
                ws.walker_spin[w, i]      = ws.new_spin[w, i]
            end
        end

        # Energy estimator
        if length(ws.E_local) < num_walkers
            resize!(ws.E_local, num_walkers)
        end
        E_sum = 0.0
        @inbounds for w in 1:num_walkers
            ws.E_local[w] = ws.new_E_local[w]
            E_sum += ws.E_local[w]
        end
        E_mean = E_sum / num_walkers
        E_var  = var(ws.E_local[1:num_walkers])
        energies[step]         = E_mean
        energies_squared[step] = E_mean^2

        # Sk(k), Sn(k) accumulation
        if step > equilibration_cutoff
            for k_idx in 1:Nk_vals
                k = (k_idx - 1) * k_min
                Sk_step = 0.0
                Sn_step = 0.0
                for w in 1:num_walkers
                    re_spin = 0.0; im_spin = 0.0
                    re_dens = 0.0; im_dens = 0.0
                    for i in 1:N_total
                        s = (ws.walker_spin[w, i] == 1.0) ? 1.0 : -1.0
                        x = ws.walker_positions[w, i]
                        re_spin += s * cos(k * x)
                        im_spin += s * sin(k * x)
                        re_dens += cos(k * x)
                        im_dens += sin(k * x)
                    end
                    Sk_step += (re_spin^2 + im_spin^2)
                    Sn_step += (re_dens^2 + im_dens^2)
                end
                Sk_acc[k_idx] += Sk_step / (num_walkers * N_total)
                Sn_acc[k_idx] += Sn_step / (num_walkers * N_total)
            end
            Sk_count += 1
            Sn_count += 1

            # Cs(r) accumulation on the same samples
            for w in 1:num_walkers
                for i in 1:N_total-1
                    si = (ws.walker_spin[w, i] == 1.0) ? 1.0 : -1.0
                    xi = ws.walker_positions[w, i]
                    for j in i+1:N_total
                        sj = (ws.walker_spin[w, j] == 1.0) ? 1.0 : -1.0
                        dx = xi - ws.walker_positions[w, j]
                        dx -= L * round(dx * invL)
                        r = abs(dx)
                        if r < L / 2
                            bin = Int(floor(r / dr)) + 1
                            if bin <= Nbins_corr
                                Cs_bins[bin] += si * sj
                                Cs_bin_count[bin] += 1
                            end
                        end
                    end
                end
            end
        end

        # Population control
        if branching
            E_T = E_mean - C_pop * log(num_walkers / num_walkers_initial)
        end

        if verbose && step % 100 == 0
            @printf("Step %d | walkers: %d | E_T = %.10f | E = %.10f | Var(E_loc) = %.10f\n",
                step, num_walkers, E_T, E_mean, E_var)
        end
    end

    start_idx          = equilibration_steps + 1
    energy_avg         = mean(energies[start_idx:end])
    energy_squared_avg = mean(energies_squared[start_idx:end])

    ### Flat file output #####################################################

    Sk_avg = Sk_count > 0 ? Sk_acc / Sk_count : zeros(Nk_vals)
    if :Sk in observables
        open("Sk.txt", "w") do f
            println(f, "# k  S_spin(k)")
            for k_idx in 1:Nk_vals
                println(f, (k_idx - 1) * k_min, " ", Sk_avg[k_idx])
            end
        end
    end

    Sn_avg = Sn_count > 0 ? Sn_acc / Sn_count : zeros(Nk_vals)
    if :Sn in observables
        open("Sn.txt", "w") do f
            println(f, "# k  S_density(k)")
            for k_idx in 1:Nk_vals
                println(f, (k_idx - 1) * k_min, " ", Sn_avg[k_idx])
            end
        end
    end

    # Cs(r) (spin correlation function), same definition as the pure estimator
    Cs_full = [Cs_bin_count[i] > 0 ? Cs_bins[i] / Cs_bin_count[i] : 0.0 for i in 1:Nbins_corr]
    if :Cs_r in observables
        open("Cs.txt", "w") do f
            println(f, "# r  Cs(r) [mixed estimator]")
            for i in 1:Nbins_corr
                r = (i - 0.5) * dr
                println(f, r, " ", Cs_full[i])
            end
        end
    end

    if save_state_path !== nothing
        save_dmc_state(save_state_path; state=(
            solver = "dmc_mixed",
            positions = ws.walker_positions[1:num_walkers, :],
            spins = ws.walker_spin[1:num_walkers, :],
            E_T = E_T,
            dt = dt,
            steps_done = num_steps,
            energies = energies,
        ))
    end

    ################################################################### error estimation
    energy_err_unc = std(energies[start_idx:end]) / sqrt(length(energies[start_idx:end]))

    if reblocking
        opt_block_size, energy_err, _, _ = run_reblocking_loop(energies[start_idx:end], 40; label="energy")
        if verbose
            println("Reblocking error estimation result:")
            println("  Optimal Block Size = ", opt_block_size)
            println("  Energy Error = ", energy_err)
        end
    else
        energy_err = energy_err_unc
    end

    elapsed_s = (time_ns() - t_start) / 1e9

    #################################################################### print results
    if verbose
        println("\n========== Final State (Last Step) ==========")
        @printf("Step         = %d\n", num_steps)
        @printf("E_last      = %.15f\n", energies[num_steps])

        println("\n========== DMC Results (Two-Component, Mixed) ==========")
        @printf("Threads      = %d\n", num_threads_to_use)
        @printf("E_DMC       = %.15f ± %.2e\n", energy_avg, energy_err)
        @printf("E_VMC       = %.15f\n", E_T_init)
        @printf("Final walkers: %d\n", num_walkers)
        @printf("Elapsed time: %.2f s (%.2f min)\n", elapsed_s, elapsed_s / 60)
        println("=================================================")
    end

    # Write scalar results
    open("results.txt", "w") do f
        @printf(f, "E_DMC        %.15f\n", energy_avg)
        @printf(f, "E_DMC_err    %.6e\n",  energy_err)
        @printf(f, "E_sq_avg     %.15f\n", energy_squared_avg)
        @printf(f, "elapsed_s    %.2f\n",   elapsed_s)
    end

    return energy_avg, energy_err, energy_squared_avg
end
