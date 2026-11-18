using Printf

# ====== SLURM settings (shared across all runs) ======
const CPUS_PER_TASK = 112
const TIME_LIMIT    = "24:00:00"
const ACCOUNT       = "lab_upc59"
const QOS           = "res_upc59_a"
const PARTITION     = "high-cpu"
const PREFIX        = "N80"

# Parallel sweep for method = "vmc_parallel": runs up to JOBS folders at once and
# redraws a status table (one row per folder) every few seconds until all are finished.
const PARALLEL_SWEEP_TEMPLATE = raw"""#!/bin/bash
cd "$(dirname "$0")"
JOBS=${1:-$(nproc)}
folders=( */ )
folders=( "${folders[@]%/}" )

trap 'trap - INT TERM; kill 0; exit 130' INT TERM

ls -d */ | xargs -P "$JOBS" -I{} bash -c 'cd "{}" && bash run_local.sh' &
sweep_pid=$!

show_status() {
    # Widen the run column to the longest label so long folder names don't shift the other columns
    local w=14
    for d in "${folders[@]}"; do
        label=$(sed -E 's/.*_(K12[0-9.]+)_.*/\1/' <<< "$d")
        (( ${#label} > w )) && w=${#label}
    done
    printf '\033[H\033[J'
    printf "%-${w}s %-16s %-10s %-10s %-12s %s\n" run progress pos_acc flip_acc E/N state
    for d in "${folders[@]}"; do
        label=$(sed -E 's/.*_(K12[0-9.]+)_.*/\1/' <<< "$d")
        log="$d/run.log"
        progress=-; pos=-; flip=-; energy=-
        if [[ -f "$log" ]]; then
            line=$(grep '^Step ' "$log" | tail -n 1)
            if [[ -n "$line" ]]; then
                read -r _ frac _ _ pos _ _ flip <<< "$line"
                progress=$(awk -F/ '{printf "%.1f%%", 100*$1/$2}' <<< "$frac")
            fi
        fi
        if [[ -f "$log" ]] && grep -q '^Measurements = ' "$log"; then
            state=done
            energy=$(awk '/^E_VMC_per_particle/ {print $2}' "$d/VMC_flips/results.txt")
        elif [[ ! -f "$log" ]]; then
            state=queued
        elif kill -0 "$sweep_pid" 2>/dev/null; then
            state=running
        else
            state=failed
        fi
        printf "%-${w}s %-16s %-10s %-10s %-12s %s\n" "$label" "$progress" "$pos" "$flip" "$energy" "$state"
    done
}

while kill -0 "$sweep_pid" 2>/dev/null; do
    show_status
    sleep 5
done
show_status
wait "$sweep_pid"
rc=$?
echo "Sweep finished (exit code $rc)."
exit $rc
"""

function generate_folders(;
    g::Float64,
    t::Float64,
    dt::Float64,
    N1::Int,
    N2::Int,
    num_steps::Int,
    vmc_production_steps::Int=80_000,
    num_walkers::Int,
    forward_walk_steps::Vector{Int},
    time_order::Int,
    h_values::Vector{Float64},
    g12_values::Vector{Float64},
    parent_folder::String="",
    acum_mixed::Bool=true,
    estimate::Bool=true,
    method::String="dmc_pure",
    measurement_stride::Int=1_000,
    equilibration_steps::Int=50_000,
    vmc_equilibration_steps::Int=20_000,
    K12_values::Vector{Float64}=Float64[])

    method in ("dmc_mixed", "dmc_pure", "vmc", "vmc_parallel", "dmc_pure_ovlp") ||
        throw(ArgumentError("method must be dmc_mixed, dmc_pure, vmc, vmc_parallel, or dmc_pure_ovlp"))
    measurement_stride > 0 || throw(ArgumentError("measurement_stride must be positive"))
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    is_vmc = method in ("vmc", "vmc_parallel")
    method_is_parallel = method == "vmc_parallel"
    !is_vmc && vmc_equilibration_steps >= num_steps &&
        throw(ArgumentError("vmc_equilibration_steps must be less than num_steps for DMC methods"))

    N_total = N1 + N2
    count = 0
    # nothing => keep the default alpha_from_a(a12) behavior for this folder set
    K12_variants = isempty(K12_values) ? [nothing] : K12_values

    for h in h_values, g12 in g12_values, K12 in K12_variants, fwd in forward_walk_steps
        folder_name = isnothing(K12) ?
            @sprintf("%s_g%.6f_g12%.6f_h%.6f_t%.6f_fwd%d_N%d", PREFIX, g, g12, h, t, fwd, N_total) :
            @sprintf("%s_g%.6f_g12%.6f_K12%.6f_h%.6f_t%.6f_fwd%d_N%d", PREFIX, g, g12, K12, h, t, fwd, N_total)
        folder = isempty(parent_folder) ? folder_name : joinpath(parent_folder, folder_name)
        mkpath(folder)
        runner_path = relpath(joinpath(@__DIR__, "run_dmc_cc_forward.jl"), abspath(folder))

        if estimate
            # MF estimate of N1, N2 based on polarization P^2 = 1 - (2t/(g12-g))^2
            denom = g12 - g
            P_mf = denom > 2*t ? sqrt(1.0 - (2*t/denom)^2) : 0.0
            N1_use = round(Int, N_total * (1 + P_mf) / 2)
            N2_use = N_total - N1_use
        else
            N1_use = N1
            N2_use = N2
        end

        # Write params.dat
        open(joinpath(folder, "params.dat"), "w") do f
            println(f, "N1                  $N1_use")
            println(f, "N2                  $N2_use")
            println(f, "method              $method")
            println(f, "measurement_stride  $measurement_stride")
            println(f, "acum_mixed         $acum_mixed")
            println(f, "equilibration_steps $equilibration_steps")
            println(f, "vmc_equilibration_steps $vmc_equilibration_steps")
            @printf(f,  "g11                 %.6f\n", g)
            @printf(f,  "g22                 %.6f\n", g)
            @printf(f,  "g12                 %.6f\n", g12)
            isnothing(K12) || @printf(f, "alpha12             %.15f\n", 1.0 / K12)
            @printf(f,  "t                   %.6f\n", t)
            @printf(f,  "h                   %.6f\n", h)
            @printf(f,  "dt                  %.6f\n", dt)
            println(f, "num_steps           $num_steps")
            println(f, "vmc_production_steps $vmc_production_steps")
            println(f, "num_walkers         $num_walkers")
            println(f, "forward_walk_steps  $fwd")
            println(f, "time_order          $time_order")
        end

        # Write job.sh
        open(joinpath(folder, "job.sh"), "w") do f
            println(f, "#!/bin/bash")
            @printf(f,  "#SBATCH --job-name=crit_exp_N%d_g12%.6f_h%.6f_t%.6f_fwd%d\n", N_total, g12, h, t, fwd)
            println(f, "#SBATCH --chdir=.")
            println(f, "#SBATCH --output=job_%j.out")
            println(f, "#SBATCH --error=job_%j.err")
            println(f, "#SBATCH --ntasks=1")
            println(f, "#SBATCH --cpus-per-task=$CPUS_PER_TASK")
            println(f, "#SBATCH --time=$TIME_LIMIT")
            println(f, "#SBATCH --partition=$PARTITION")
            println(f, "#SBATCH --account=$ACCOUNT")
            println(f, "#SBATCH --qos=$QOS")
            println(f, "")
            println(f, "module load modulepath/EESSI/2025.06")
            println(f, "module load Julia/1.12.2")
            println(f, "julia --project=~ -t \$SLURM_CPUS_PER_TASK $runner_path 2>&1 | tee run.log")
        end

        # vmc_parallel: one thread per process, output only to run.log (the sweep shows the status table)
        open(joinpath(folder, "run_local.sh"), "w") do f
            println(f, "#!/bin/bash")
            if method_is_parallel
                println(f, "export OPENBLAS_NUM_THREADS=1")
                println(f, "julia -t 1 $runner_path > run.log 2>&1")
            else
                println(f, "set -o pipefail")
                println(f, "julia -t auto $runner_path 2>&1 | tee run.log")
            end
        end

        println("Created: $folder")
        count += 1
    end

    if !isempty(parent_folder)
        sweep_path = joinpath(parent_folder, "run_sweep.sh")
        open(sweep_path, "w") do f
            if method_is_parallel
                write(f, PARALLEL_SWEEP_TEMPLATE)
            else
                println(f, "#!/bin/bash")
                println(f, "set -e")
                println(f, "cd \"\$(dirname \"\$0\")\"")
                println(f, "for run_script in */run_local.sh; do")
                println(f, "    [[ -f \"\$run_script\" ]] || continue")
                println(f, "    run_folder=\${run_script%/run_local.sh}")
                println(f, "    printf 'Running %s\\n' \"\$run_folder\"")
                println(f, "    (cd \"\$run_folder\" && bash run_local.sh)")
                println(f, "done")
            end
        end
        chmod(sweep_path, 0o755)
    end

    return count
end

if abspath(PROGRAM_FILE) == @__FILE__

    hmin = 0.1e-4
    hmax = 1.0e-4
    number_of_h = 20
    h_values = collect(range(hmin, hmax, length=number_of_h))
    h_values = [0.0]

    dt = 0.025
    num_steps = 600_000
    num_walkers = 1000
    time_order = 2
    method = "dmc_mixed"
    measurement_stride = 1_000
    equilibration_steps = 150_000
    vmc_equilibration_steps = 200_000
    vmc_production_steps = 700_000
    estimate = true
    parent_folder = PREFIX
    acum_mixed = true
    sizes = [(40,40)]
    g_values = [0.01]
    t_fracs = [0.125]
    forward_walk_steps = [30_000]

    local total = 0
    for g in g_values, t_frac in t_fracs
        t = t_frac * g

            
            g12MFc = g + 2*t
            g12min = 0.95*g12MFc
            g12max = 1.15*g12MFc

            Npoints = 25
            step = (g12max-g12min)/Npoints


        g12_values = collect(g12min:step:g12max)

        g12_values = [0.5*g12MFc, 2*g12MFc, 1.05*g12MFc]
        g12_values = [0.012694, 0.005, 0.02]

        K12min = 25.
        K12max = 28.
        nkmax = 30

        K12_values = collect(K12min:((K12max-K12min)/nkmax):K12max)
        K12_values = []


        
        for (N1, N2) in sizes
            total += generate_folders(;
                g=g,
                t=t,
                dt=dt,
                N1=N1,
                N2=N2,
                num_steps=num_steps,
                vmc_production_steps=vmc_production_steps,
                num_walkers=num_walkers,
                forward_walk_steps=forward_walk_steps,
                time_order=time_order,
                h_values=h_values,
                g12_values=g12_values,
                parent_folder=parent_folder,
                acum_mixed=acum_mixed,
                estimate=estimate,
                method=method,
                measurement_stride=measurement_stride,
                equilibration_steps=equilibration_steps,
                vmc_equilibration_steps=vmc_equilibration_steps,
                # K12_values=K12_values
                )
        end
    end

    println("Total folders created: $total")
end

