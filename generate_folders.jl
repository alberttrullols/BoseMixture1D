using Printf

include(joinpath(@__DIR__, "constants.jl"))
include(joinpath(@__DIR__, "machines.jl"))

const CPUS_PER_TASK = 112
const TIME_LIMIT    = "24:00:00"
const RUNNER_NAME   = "run_simulation.jl"  

const DENSITY = 1.0 


const PARALLEL_SWEEP_TEMPLATE_PATH = joinpath(@__DIR__, "templates", "run_sweep_parallel.sh")

function enabled_observables(observables::Dict{Symbol, Bool})
    return sort([k for (k, v) in observables if v])
end

function validate_choices(; method, trial_wf, observables)
    method in METHODS ||
        throw(ArgumentError("method must be one of $(join(METHODS, ", "))"))
    trial_wf in TRIAL_WFS ||
        throw(ArgumentError("trial_wf must be one of $(join(TRIAL_WFS, ", "))"))
    allowed = union(values(SUPPORTED_OBSERVABLES)...)
    for k in keys(observables)
        k in allowed || throw(ArgumentError("unknown observable $k"))
    end
    supported = SUPPORTED_OBSERVABLES[method]
    for k in keys(observables)
        observables[k] || continue
        k in supported || throw(ArgumentError(
            "observable $k is not produced by $method (supported: $(join(sort(collect(supported)), ", ")))"))
    end
    return nothing
end

function generate_folders(;
    g::Float64,
    dt::Float64,
    size_pairs::Vector{Tuple{Int, Int}},
    num_steps::Int,
    num_walkers::Int,
    forward_walk_steps_values::Vector{Int},
    time_order::Int,
    g12_values::Vector{Float64},
    method::String,
    trial_wf::String,
    observables::Dict{Symbol, Bool},
    prefix::String,
    parent_folder::String="",
    acum_mixed::Bool=true,
    measurement_stride::Int=1_000,
    equilibration_steps::Int=50_000,
    vmc_equilibration_steps::Int=20_000,
    vmc_production_steps::Int=80_000,
    vmc_step_size::Float64=4.0,
    save_state::Bool=false,
    R_match_same_values::Vector{Float64}=Float64[],
    R_match_ud_values::Vector{Float64}=Float64[])

    validate_choices(; method, trial_wf, observables)
    vmc_step_size > 0 || throw(ArgumentError("vmc_step_size must be positive"))
    !isempty(size_pairs) && all(p -> p[1] >= 0 && p[2] >= 0 && p[1] + p[2] > 0, size_pairs) ||
        throw(ArgumentError("size_pairs must be a nonempty list of (N1, N2) with nonnegative entries and N1+N2 > 0"))
    measurement_stride > 0 || throw(ArgumentError("measurement_stride must be positive"))
    equilibration_steps >= 0 || throw(ArgumentError("equilibration_steps must be nonnegative"))
    !isempty(forward_walk_steps_values) && all(>(0), forward_walk_steps_values) ||
        throw(ArgumentError("forward_walk_steps_values must be a nonempty list of positive integers"))
    if !isempty(R_match_same_values) || !isempty(R_match_ud_values)
        trial_wf == TRIAL_2B_UNCOUPLED_MATCHING ||
            throw(ArgumentError("R_match_same_values/R_match_ud_values need trial_wf=$TRIAL_2B_UNCOUPLED_MATCHING"))
        all(>(0), R_match_same_values) && all(>(0), R_match_ud_values) ||
            throw(ArgumentError("R_match_same_values and R_match_ud_values must be positive"))
    end

    is_vmc = method in (VMC, VMC_PARALLEL)
    method_is_parallel = method == VMC_PARALLEL
    !is_vmc && equilibration_steps >= num_steps &&
        throw(ArgumentError("equilibration_steps must be less than num_steps for DMC methods"))

    obs_list = enabled_observables(observables)
    obs_string = isempty(obs_list) ? "none" : join(obs_list, ",")
    count = 0
    # nothing => keep the default exponents from g12 and g for this folder set
    R_same_variants = isempty(R_match_same_values) ? [nothing] : R_match_same_values
    R_ud_variants = isempty(R_match_ud_values) ? [nothing] : R_match_ud_values

    for (N1, N2) in size_pairs, g12 in g12_values, fwd in forward_walk_steps_values,
        R_same in R_same_variants, R_ud in R_ud_variants
        N_total = N1 + N2
        L = N_total / DENSITY
        # Matching radii default to L/2 (the exact solution); written explicitly so each folder is self-describing.
        R_params = Pair{String, Float64}[]
        R_tag = ""
        if trial_wf == TRIAL_2B_UNCOUPLED_MATCHING
            R_same_use = isnothing(R_same) ? L / 2 : R_same
            R_ud_use = isnothing(R_ud) ? L / 2 : R_ud
            all(R -> 0.0 < R <= L / 2, (R_same_use, R_ud_use)) ||
                throw(ArgumentError("matching radii must satisfy 0 < R <= L/2 = $(L / 2), got R_match_same=$R_same_use R_match_ud=$R_ud_use"))
            push!(R_params, "R_match_same" => R_same_use, "R_match_ud" => R_ud_use)
            R_tag = @sprintf("_Rs%.6f_Rud%.6f", R_same_use, R_ud_use)
        end
        size_tag = @sprintf("_N1_%d_N2_%d", N1, N2)
        folder_name = @sprintf("%s_%s_g%.6f_g12%.6f_fwd%d%s_nw%d",
                               prefix, trial_wf, g, g12, fwd, size_tag, num_walkers) * R_tag
        folder = isempty(parent_folder) ? folder_name : joinpath(parent_folder, folder_name)
        mkpath(folder)
        runner_path = relpath(joinpath(@__DIR__, RUNNER_NAME), abspath(folder))
        project_path = dirname(runner_path)

        open(joinpath(folder, "params.dat"), "w") do f
            println(f, "N1                  $N1")
            println(f, "N2                  $N2")
            println(f, "method              $method")
            println(f, "trial_wf            $trial_wf")
            println(f, "observables         $obs_string")
            println(f, "L                   $L")
            println(f, "measurement_stride  $measurement_stride")
            println(f, "acum_mixed          $acum_mixed")
            println(f, "equilibration_steps $equilibration_steps")
            println(f, "vmc_equilibration_steps $vmc_equilibration_steps")
            println(f, "vmc_production_steps $vmc_production_steps")
            println(f, "vmc_step_size       $vmc_step_size")
            println(f, "save_state          $save_state")
            @printf(f,  "g11                 %.6f\n", g)
            @printf(f,  "g22                 %.6f\n", g)
            @printf(f,  "g12                 %.6f\n", g12)
            for (key, val) in R_params
                @printf(f, "%-20s%.12f\n", key, val)
            end
            @printf(f,  "dt                  %.6f\n", dt)
            println(f, "num_steps           $num_steps")
            println(f, "num_walkers         $num_walkers")
            println(f, "forward_walk_steps  $fwd")
            println(f, "time_order          $time_order")
        end

        open(joinpath(folder, "job.sh"), "w") do f
            println(f, "#!/bin/bash")
            @printf(f,  "#SBATCH --job-name=%s%s_g12%.6f_fwd%d\n", trial_wf, size_tag, g12, fwd)
            println(f, "#SBATCH --chdir=.")
            println(f, "#SBATCH --output=job_%j.out")
            println(f, "#SBATCH --error=job_%j.err")
            println(f, "#SBATCH --ntasks=1")
            println(f, "#SBATCH --cpus-per-task=$CPUS_PER_TASK")
            println(f, "#SBATCH --time=$TIME_LIMIT")
            println(f, "#SBATCH --account=$(MACHINE.account)")
            println(f, "#SBATCH --qos=$(MACHINE.qos)")
            println(f, "")
            foreach(line -> println(f, line), MACHINE.modules)
            println(f, "julia --project=\"$project_path\" -t \$SLURM_CPUS_PER_TASK \"$runner_path\" 2>&1 | tee run.log")
        end

        open(joinpath(folder, "run_local.sh"), "w") do f
            println(f, "#!/bin/bash")
            if method_is_parallel
                println(f, "export OPENBLAS_NUM_THREADS=1")
                println(f, "julia --project=\"$project_path\" -t 1 \"$runner_path\" > run.log 2>&1")
            else
                println(f, "set -o pipefail")
                println(f, "julia --project=\"$project_path\" -t auto \"$runner_path\" 2>&1 | tee run.log")
            end
        end

        println("Created: $folder")
        count += 1
    end

    if !isempty(parent_folder)
        sweep_path = joinpath(parent_folder, "run_sweep.sh")
        open(sweep_path, "w") do f
            if method_is_parallel
                write(f, read(PARALLEL_SWEEP_TEMPLATE_PATH, String))
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
