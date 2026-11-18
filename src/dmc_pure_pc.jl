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

# Pre-allocated workspace (Float64 spin)
mutable struct DMCWorkspace_2comp_pure
    walker_positions::Matrix{Float64} # (num_walkers, N_total)
    walker_spin::Matrix{Float64}      # (num_walkers, N_total): 1.0 or 2.0, fixed species label
    new_positions::Matrix{Float64}    # buffer for next generation
    new_spin::Matrix{Float64}         # buffer for spin after branching
    M::Vector{Int}                    # copies per walker
    E_local::Vector{Float64}          # local energy per walker
    new_E_local::Vector{Float64}      # replicated local energies after branching

    # Pure (forward-walking) coordinates
    pure_positions::Matrix{Float64}     # Snapshot of positions at block start
    pure_spin::Matrix{Float64}          # Snapshot of spins at block start
    new_pure_positions::Matrix{Float64} # Buffer for branching
    new_pure_spin::Matrix{Float64}      # Buffer for branching
end

function DMCWorkspace_2comp_pure(num_walkers::Int, N_total::Int)
    max_walkers = num_walkers * MAX_COPIES  # worst case
    return DMCWorkspace_2comp_pure(
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Vector{Int}(undef, num_walkers),
        Vector{Float64}(undef, num_walkers),
        Vector{Float64}(undef, max_walkers),

        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, num_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total),
        Matrix{Float64}(undef, max_walkers, N_total)
    )
end

function ancestor_stats(ancestor::Vector{Int})
    counts = Dict{Int,Int}()
    for a in ancestor
        counts[a] = get(counts, a, 0) + 1
    end
    W = length(ancestor)
    return length(counts), W^2 / sum(c -> c^2, values(counts))
end

"""
Pure (forward-walking) Diffusion Monte Carlo for 1D two-component bosons with contact
interactions, with population control. Species labels are fixed for the whole run (no
inter-species tunneling), so the only per-walker observable propagated through branching
is the local energy. Besides the usual mixed estimator (current-population average at
every step), a projected ("pure") estimator is reconstructed every `forward_walk_steps`
steps by snapshotting walker configurations and following their surviving descendants,
giving unbiased estimates of operators that do not commute with H (Cs(r), Sk(k), Sn(k),
and - for cross-validation - the local energy itself).
"""
function DMC_pure_pc(num_steps::Int, num_walkers::Int, dt::Float64, E_T_init::Float64,
    tab::PairTrial,
    N1::Int, N2::Int;
    seed::Int=1234,
    importance_sampling::Bool=true,
    branching::Bool=true,
    time_order::Int=1,
    verbose::Bool=true,
    reblocking::Bool=true,
    num_threads_to_use::Int=Threads.nthreads(),
    forward_walk_steps::Int=100,
    acum_mixed::Bool=true,
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
    C_pop = 1.0 / dt    # population control constant
    energies = zeros(Float64, num_steps)
    energies_squared = zeros(Float64, num_steps)

    equilibration_cutoff = equilibration_steps

    k_min = 2π / L
    Nk_vals = 75  # number of k values to compute
    Nbins_corr = 100
    dr = (L / 2) / Nbins_corr

    ws = DMCWorkspace_2comp_pure(num_walkers, N_total)
    ancestor = collect(1:num_walkers)  # snapshot index each walker descends from

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
    x_buffers         = [zeros(N_total) for _ in 1:n_buffers]
    drift_buffers     = [zeros(N_total) for _ in 1:n_buffers]
    drift_old_buffers = [zeros(N_total) for _ in 1:n_buffers]
    rand_buffers      = [zeros(N_total) for _ in 1:n_buffers]
    spin_buffers      = [zeros(Float64, N_total) for _ in 1:n_buffers]
    rng_threads       = [MersenneTwister(seed + tid - 1) for tid in 1:n_buffers]

    # Thread-local structure factor and correlation buffers (pure estimators)
    Sk_threads       = [zeros(Float64, Nk_vals) for _ in 1:n_buffers]
    Sn_threads       = [zeros(Float64, Nk_vals) for _ in 1:n_buffers]
    Cs_threads       = [zeros(Float64, Nbins_corr) for _ in 1:n_buffers]
    Cs_count_threads = [zeros(Int, Nbins_corr) for _ in 1:n_buffers]

    block_size = forward_walk_steps
    pure_block_count = 0

    Sk_acc_pure = zeros(Float64, Nk_vals)
    Sn_acc_pure = zeros(Float64, Nk_vals)
    Cs          = zeros(Float64, Nbins_corr)
    Cs_count    = zeros(Int, Nbins_corr)

    energy_series_pure = Float64[]

    anc_steps   = Int[]
    anc_walkers = Int[]
    anc_counts  = Int[]
    anc_neff    = Float64[]

    #################################################################### dmc loop
    for step in 1:num_steps

        if (step - 1) % block_size == 0
            if size(ws.pure_positions, 1) < num_walkers
                ws.pure_positions = Matrix{Float64}(undef, num_walkers, N_total)
                ws.pure_spin = Matrix{Float64}(undef, num_walkers, N_total)
            end

            # Copy current state to the historical snapshot
            ancestor = collect(1:num_walkers)
            @inbounds for w in 1:num_walkers
                for i in 1:N_total
                    ws.pure_positions[w, i] = ws.walker_positions[w, i]
                    ws.pure_spin[w, i] = ws.walker_spin[w, i]
                end
            end
        end

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
                xb[i] = ws.walker_positions[w, i]
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
                    xb[i] = mod(xb[i] +
                                randb[i] * diffusion_scale, L)
                end
            end

            # Compute drift and energy at new position
            E_new = turbo_compute_drift_and_energy!(F_new, xb, spin_buf, tab; N_total, L, invL)

            ws.E_local[w] = E_new

            # Branching
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
                ws.walker_spin[w, i] = spin_buf[i]
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

        # Ensure new_pure arrays are sized correctly
        if size(ws.new_pure_positions, 1) < total_new
            ws.new_pure_positions = Matrix{Float64}(undef, total_new, N_total)
            ws.new_pure_spin = Matrix{Float64}(undef, total_new, N_total)
        end

        new_ancestor = Vector{Int}(undef, total_new)
        idx = 1
        @inbounds for w in 1:num_walkers
            for _ in 1:ws.M[w]
                for i in 1:N_total
                    ws.new_positions[idx, i] = ws.walker_positions[w, i]
                    ws.new_spin[idx, i] = ws.walker_spin[w, i]

                    # Replicate the pure coordinates
                    ws.new_pure_positions[idx, i] = ws.pure_positions[w, i]
                    ws.new_pure_spin[idx, i] = ws.pure_spin[w, i]
                end
                ws.new_E_local[idx] = ws.E_local[w]
                new_ancestor[idx] = ancestor[w]
                idx += 1
            end
        end

        num_walkers = total_new
        ancestor = new_ancestor

        if size(ws.walker_positions, 1) < num_walkers
            ws.walker_positions = Matrix{Float64}(undef, num_walkers, N_total)
        end
        if size(ws.walker_spin, 1) < num_walkers
            ws.walker_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        # Ensure main pure arrays are sized correctly
        if size(ws.pure_positions, 1) < num_walkers
            ws.pure_positions = Matrix{Float64}(undef, num_walkers, N_total)
            ws.pure_spin = Matrix{Float64}(undef, num_walkers, N_total)
        end

        @inbounds for w in 1:num_walkers
            for i in 1:N_total
                ws.walker_positions[w, i] = ws.new_positions[w, i]
                ws.walker_spin[w, i] = ws.new_spin[w, i]

                # Finalize the pure coordinates for the next step
                ws.pure_positions[w, i] = ws.new_pure_positions[w, i]
                ws.pure_spin[w, i] = ws.new_pure_spin[w, i]
            end
        end

        # Energy estimator (mixed: current-population average)
        if length(ws.E_local) < num_walkers
            resize!(ws.E_local, num_walkers)
        end
        E_sum = 0.0
        @inbounds for w in 1:num_walkers
            ws.E_local[w] = ws.new_E_local[w]
            E_sum += ws.E_local[w]
        end

        E_mean = E_sum / num_walkers
        E_var = var(ws.E_local[1:num_walkers])
        energies[step] = E_mean
        energies_squared[step] = E_mean^2

        # histo, Cs, Sk, Sn calculations (pure, forward-walked estimators)
        if step > equilibration_cutoff && step % block_size == 0

            # Pure (forward-walked) energy estimator at the block-start snapshot
            if acum_mixed
                E_pure_w = zeros(Float64, num_walkers)
                Threads.@threads for w in 1:num_walkers
                    tid = Threads.threadid()
                    xb_pure = x_buffers[tid]
                    spin_pure = spin_buffers[tid]
                    F_pure = drift_buffers[tid]
                    @inbounds for i in 1:N_total
                        xb_pure[i] = ws.pure_positions[w, i]
                        spin_pure[i] = ws.pure_spin[w, i]
                    end
                    E_pure_w[w] = turbo_compute_drift_and_energy!(F_pure, xb_pure, spin_pure, tab; N_total, L, invL)
                end
                push!(energy_series_pure, sum(E_pure_w) / num_walkers)
            end

            # Pure Structure Factor (spin and density) and spin correlation
            for tid in 1:num_threads_to_use
                fill!(Sk_threads[tid], 0.0)
                fill!(Sn_threads[tid], 0.0)
                fill!(Cs_threads[tid], 0.0)
                fill!(Cs_count_threads[tid], 0)
            end

            Threads.@threads for w in 1:num_walkers
                tid = Threads.threadid()
                # Sk, Sn
                for k_idx in 1:Nk_vals
                    k = (k_idx - 1) * k_min
                    re_spin = 0.0; im_spin = 0.0
                    re_dens = 0.0; im_dens = 0.0
                    for i in 1:N_total
                        s = (ws.pure_spin[w, i] == 1.0) ? 1.0 : -1.0
                        x = ws.pure_positions[w, i]
                        re_spin += s * cos(k * x)
                        im_spin += s * sin(k * x)
                        re_dens += cos(k * x)
                        im_dens += sin(k * x)
                    end
                    Sk_threads[tid][k_idx] += (re_spin^2 + im_spin^2) / (num_walkers * N_total)
                    Sn_threads[tid][k_idx] += (re_dens^2 + im_dens^2) / (num_walkers * N_total)
                end
                # Cs(r)
                for i in 1:N_total-1
                    si = (ws.pure_spin[w, i] == 1.0) ? 1.0 : -1.0
                    xi = ws.pure_positions[w, i]
                    for j in i+1:N_total
                        sj = (ws.pure_spin[w, j] == 1.0) ? 1.0 : -1.0
                        xj = ws.pure_positions[w, j]
                        dx = xi - xj
                        dx -= L * round(dx * invL)
                        r = abs(dx)
                        if r < L / 2
                            bin = Int(floor(r / dr)) + 1
                            if bin <= Nbins_corr
                                Cs_threads[tid][bin]       += si * sj
                                Cs_count_threads[tid][bin] += 1
                            end
                        end
                    end
                end
            end

            for tid in 1:num_threads_to_use
                Sk_acc_pure .+= Sk_threads[tid]
                Sn_acc_pure .+= Sn_threads[tid]
                Cs          .+= Cs_threads[tid]
                Cs_count    .+= Cs_count_threads[tid]
            end
            pure_block_count += 1
        end

        # Population control
        if branching
            E_T = E_mean - C_pop * log(num_walkers / num_walkers_initial)
        end

        if step % 100 == 0
            n_anc, N_eff = ancestor_stats(ancestor)
            push!(anc_steps, step)
            push!(anc_walkers, num_walkers)
            push!(anc_counts, n_anc)
            push!(anc_neff, N_eff)
            if verbose
                @printf("Step %d | walkers: %d | ancestors: %d | N_eff: %.1f | E_T = %.10f | E = %.10f | Var(E_loc) = %.10f\n",
                    step, num_walkers, n_anc, N_eff, E_T, E_mean, E_var)
            end
        end
    end

    # Calculate final pure estimators
    if pure_block_count > 0
        Sk_avg_pure = Sk_acc_pure / pure_block_count
        Sn_avg_pure = Sn_acc_pure / pure_block_count
        Cs_full     = [Cs_count[i] > 0 ? Cs[i] / Cs_count[i] : 0.0 for i in 1:Nbins_corr]
    else
        Sk_avg_pure = zeros(Nk_vals)
        Sn_avg_pure = zeros(Nk_vals)
        Cs_full     = zeros(Nbins_corr)
    end

    if acum_mixed && length(energy_series_pure) > 0
        energy_pure_avg = mean(energy_series_pure)
        energy_pure_err = length(energy_series_pure) > 1 ?
            std(energy_series_pure) / sqrt(length(energy_series_pure)) : 0.0
    else
        energy_pure_avg = NaN
        energy_pure_err = NaN
    end

    start_idx          = equilibration_steps + 1
    energy_avg         = mean(energies[start_idx:end])
    energy_squared_avg = mean(energies_squared[start_idx:end])

    ### Saving function files

    # Write Sk at all k values (spin structure factor, pure estimator)
    if :Sk in observables
        open("Sk.txt", "w") do f
            println(f, "# k values and corresponding S_spin(k) [pure estimator]")
            for k_idx in 1:Nk_vals
                k = (k_idx - 1) * k_min
                println(f, k, " ", Sk_avg_pure[k_idx])
            end
        end
    end

    # Write Sn at all k values (density structure factor, pure estimator)
    if :Sn in observables
        open("Sn.txt", "w") do f
            println(f, "# k values and corresponding S_density(k) [pure estimator]")
            for k_idx in 1:Nk_vals
                k = (k_idx - 1) * k_min
                println(f, k, " ", Sn_avg_pure[k_idx])
            end
        end
    end

    # Write Cs(r) (spin correlation function, pure estimator)
    if :Cs_r in observables
        open("Cs.txt", "w") do f
            println(f, "# r  Cs(r) [pure estimator]")
            for i in 1:Nbins_corr
                r = (i - 0.5) * dr
                println(f, r, " ", Cs_full[i])
            end
        end
    end

    # Write energy stream (one value per step, after equilibration) - mixed estimator
    open("energy_stream.txt", "w") do f
        println(f, "# step  E_mean")
        for i in start_idx:num_steps
            println(f, i, " ", energies[i])
        end
    end

    # Write pure (forward-walked) energy stream (one value per block)
    open("energy_pure_stream.txt", "w") do f
        println(f, acum_mixed ? "# block  <E>_pure" : "# not accumulated (acum_mixed=false)")
        if acum_mixed
            for b in eachindex(energy_series_pure)
                println(f, b, " ", energy_series_pure[b])
            end
        end
    end

    # Write ancestor diagnostics (every 100 steps)
    open("ancestor_stats.txt", "w") do f
        println(f, "# step  walkers  n_ancestors  N_eff")
        for i in eachindex(anc_steps)
            println(f, anc_steps[i], " ", anc_walkers[i], " ", anc_counts[i], " ", anc_neff[i])
        end
    end

    if save_state_path !== nothing
        save_dmc_state(save_state_path; state=(
            solver = "dmc_pure_pc",
            positions = ws.walker_positions[1:num_walkers, :],
            spins = ws.walker_spin[1:num_walkers, :],
            ancestor = ancestor,
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

    #################################################################### print stuff
    elapsed_s = (time_ns() - t_start) / 1e9
    if verbose
        final_var = var(ws.E_local[1:num_walkers])

        # Final state from last step (averaged over all walkers)
        println("\n========== Final State (Last Step) ==========")
        @printf("Step         = %d\n", num_steps)
        @printf("E_last      = %.15f\n", energies[num_steps])

        println("\n========== DMC Results (Two-Component, Pure) ==========")
        @printf("Threads      = %d\n", num_threads_to_use)
        @printf("E_DMC       = %.15f ± %.2e\n", energy_avg, energy_err)
        @printf("E_VMC       = %.15f\n", E_T_init)
        if acum_mixed
            @printf("E_pure      = %.15f ± %.2e\n", energy_pure_avg, energy_pure_err)
        else
            println("Pure (forward-walked) energy estimator: not accumulated")
        end
        @printf("Final walkers: %d\n", num_walkers)
        @printf("Elapsed time: %.2f s (%.2f min)\n", elapsed_s, elapsed_s / 60)
        println("=================================================")
    end

    # Write final scalar results to file
    open("results.txt", "w") do f
        @printf(f, "acum_mixed   %s\n", acum_mixed)
        @printf(f, "E_DMC        %.15f\n", energy_avg)
        @printf(f, "E_DMC_err    %.6e\n",  energy_err)
        @printf(f, "E_sq_avg     %.15f\n", energy_squared_avg)
        @printf(f, "E_pure_avg   %.15f\n", energy_pure_avg)
        @printf(f, "E_pure_err   %.6e\n",  energy_pure_err)
        @printf(f, "elapsed_s    %.2f\n",   elapsed_s)
    end

    return energy_avg, energy_err, energy_squared_avg, energy_pure_avg, energy_pure_err
end
