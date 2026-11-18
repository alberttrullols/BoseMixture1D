using Printf

const SRC_DIR = joinpath(@__DIR__, "src")

const TWO_BODY_TRIALS = ("2b_uncoupled", "2b_uncoupled_matching", "2b_phonons")
const SUPPORTED = Set([
    (method, trial)
    for method in ("vmc", "vmc_parallel", "dmc_mixed", "dmc_pure", "dmc_pure_ovlp")
    for trial in TWO_BODY_TRIALS
])

# Mirrors SUPPORTED_OBSERVABLES in generate_folders.jl.
const VMC_OBSERVABLES = Set([:Cs_r, :Cn_r, :Sk, :Sn])
const DMC_OBSERVABLES = Set([:Cs_r, :Sk, :Sn])
const OBSERVABLES_BY_METHOD = Dict(
    "vmc"           => VMC_OBSERVABLES,
    "vmc_parallel"  => VMC_OBSERVABLES,
    "dmc_mixed"     => DMC_OBSERVABLES,
    "dmc_pure"      => DMC_OBSERVABLES,
    "dmc_pure_ovlp" => DMC_OBSERVABLES,
)

function read_params(filename::String="params.dat")
    params = Dict{String, String}()
    open(filename) do f
        for line in eachline(f)
            line = strip(line)
            isempty(line) && continue
            startswith(line, "#") && continue
            parts = split(line)
            length(parts) >= 2 && (params[parts[1]] = parts[2])
        end
    end
    return params
end


function load_solvers()
    include(joinpath(SRC_DIR, "core.jl"))
    include(joinpath(SRC_DIR, "reblocking.jl"))
    include(joinpath(SRC_DIR, "state_io.jl"))
    include(joinpath(SRC_DIR, "vmc.jl"))
    include(joinpath(SRC_DIR, "dmc_mixed.jl"))
    include(joinpath(SRC_DIR, "dmc_pure_pc.jl"))
    include(joinpath(SRC_DIR, "dmc_pure_pc_overlapping.jl"))
end

function build_2b_trial(p::Dict{String, String}, L::Float64)
    g11 = parse(Float64, p["g11"])
    g22 = parse(Float64, p["g22"])
    g12 = parse(Float64, p["g12"])
    trial_wf = p["trial_wf"]
    trial_wf in TWO_BODY_TRIALS || error("unknown two-body trial_wf=$trial_wf")
    # Sector order (11, 12, 22) as in pair_sector; a = -2 hbar^2/(m g) with hbar = m = 1.
    a = (-2.0 / g11, -2.0 / g12, -2.0 / g22)
    if trial_wf == "2b_uncoupled_matching"
        # Matching radii for the 11/22 sector (shared, since g11 == g22) and the 12 sector; default L/2 recovers 2b_uncoupled.
        R_same = haskey(p, "R_match_same") ? parse(Float64, p["R_match_same"]) : L / 2
        R_ud = haskey(p, "R_match_ud") ? parse(Float64, p["R_match_ud"]) : L / 2
        all(R -> 0.0 < R <= L / 2, (R_same, R_ud)) ||
            error("matching radii must satisfy 0 < R <= L/2 = $(L / 2), got R_match_same=$R_same R_match_ud=$R_ud")
        tab = Base.invokelatest(pair_matched, a, (R_same, R_ud, R_same))
    elseif trial_wf == "2b_phonons"
        # Short-range matched piece glued to the long-distance Luttinger-liquid phonon tail;
        # alpha = 1/K defaults to the physical value derived from each sector's scattering
        # length, but can be overridden per sector (e.g. for a K12 sweep).
        a11, a12, a22 = a
        alpha11 = haskey(p, "alpha11") ? parse(Float64, p["alpha11"]) : Base.invokelatest(alpha_from_a, a11)
        alpha22 = haskey(p, "alpha22") ? parse(Float64, p["alpha22"]) : Base.invokelatest(alpha_from_a, a22)
        alpha12 = haskey(p, "alpha12") ? parse(Float64, p["alpha12"]) : Base.invokelatest(alpha_from_a, a12)
        @printf("Effective Luttinger parameters: K11 = %.6f (alpha11 = %.6f), K22 = %.6f (alpha22 = %.6f), K12 = %.6f (alpha12 = %.6f)\n",
            1.0 / alpha11, alpha11, 1.0 / alpha22, alpha22, 1.0 / alpha12, alpha12)
        Base.invokelatest(warn_missing_matching_point, "alpha11", alpha11, a11, L)
        Base.invokelatest(warn_missing_matching_point, "alpha22", alpha22, a22, L)
        Base.invokelatest(warn_missing_matching_point, "alpha12", alpha12, a12, L)
        tab = Base.invokelatest(PairPhonon, a, (alpha11, alpha12, alpha22), L)
    else
        tab = Base.invokelatest(PairCosine, a, L)
    end
    return tab
end

function run_vmc_2b_coupled(p::Dict{String, String})
    N1 = parse(Int, p["N1"])
    N2 = parse(Int, p["N2"])
    L = parse(Float64, get(p, "L", "0.0"))
    equilibration_steps = parse(Int, p["vmc_equilibration_steps"])
    production_steps = parse(Int, p["vmc_production_steps"])
    step_size = parse(Float64, get(p, "vmc_step_size", "4.0"))
    L == 0.0 && (L = Float64(N1 + N2))

    load_solvers()
    tab = Base.invokelatest(build_2b_trial, p, L)
    return Base.invokelatest(run_metropolis, tab;
        L=L, N1=N1, N2=N2,
        equilibration_steps=equilibration_steps,
        production_steps=production_steps, step_size=step_size, verbose=true)
end

function run_dmc_2b_coupled(p::Dict{String, String}, method::String)
    N1 = parse(Int, p["N1"])
    N2 = parse(Int, p["N2"])
    N_total = N1 + N2
    # The DMC solvers set L = N_total internally, so the trial must use the same L.
    L_param = parse(Float64, get(p, "L", "0.0"))
    (L_param == 0.0 || L_param == N_total) ||
        throw(ArgumentError("DMC uses L = N1 + N2 = $N_total; got L = $L_param"))
    dt = parse(Float64, p["dt"])
    num_steps = parse(Int, p["num_steps"])
    num_walkers = parse(Int, p["num_walkers"])
    time_order = parse(Int, p["time_order"])
    equilibration_steps = parse(Int, get(p, "equilibration_steps", "0"))
    measurement_stride = parse(Int, get(p, "measurement_stride", "1000"))
    acum_mixed = parse(Bool, get(p, "acum_mixed", "true"))
    save_state_path = parse(Bool, get(p, "save_state", "false")) ? "final_state.jls" : nothing

    load_solvers()
    tab = Base.invokelatest(build_2b_trial, p, Float64(N_total))

    # E_T from VMC
    vmc_equilibration_steps = parse(Int, p["vmc_equilibration_steps"])
    vmc_production_steps = parse(Int, p["vmc_production_steps"])
    vmc_step_size = parse(Float64, get(p, "vmc_step_size", "4.0"))
    E_T_init, _ = Base.invokelatest(run_metropolis, tab;
        N1=N1, N2=N2, equilibration_steps=vmc_equilibration_steps,
        production_steps=vmc_production_steps, step_size=vmc_step_size,
        verbose=true)

    if method == "dmc_mixed"
        observables = parse_observables(p, method)
        return Base.invokelatest(DMC_mixed, num_steps, num_walkers, dt, E_T_init, tab, N1, N2;
            importance_sampling=true, branching=true, time_order=time_order, verbose=true,
            reblocking=true, equilibration_steps=equilibration_steps, seed=1234,
            observables=observables, save_state_path=save_state_path)
    end

    forward_walk_steps = parse(Int, p["forward_walk_steps"])
    observables = parse_observables(p, method)
    dmc_kwargs = (
        importance_sampling=true, branching=true, time_order=time_order, verbose=true,
        reblocking=true,
        forward_walk_steps=forward_walk_steps, acum_mixed=acum_mixed,
        equilibration_steps=equilibration_steps, seed=1234,
        observables=observables, save_state_path=save_state_path,
    )
    if method == "dmc_pure_ovlp"
        return Base.invokelatest(DMC_pure_pc_overlapping, num_steps, num_walkers, dt, E_T_init,
            tab, N1, N2; dmc_kwargs..., measurement_stride=measurement_stride)
    end
    return Base.invokelatest(DMC_pure_pc, num_steps, num_walkers, dt, E_T_init,
        tab, N1, N2; dmc_kwargs...)
end

function parse_observables(p::Dict{String, String}, method::String)
    raw = get(p, "observables", "none")
    raw == "none" && return Set{Symbol}()
    supported = OBSERVABLES_BY_METHOD[method]
    selected = Set{Symbol}()
    for name in split(raw, ",")
        obs = Symbol(name)
        obs in supported || throw(ArgumentError(
            "observable $obs is not produced by $method (supported: $(join(sort(collect(supported)), ", ")))"))
        push!(selected, obs)
    end
    return selected
end

function main(filename::String="params.dat")
    p = read_params(filename)
    method = p["method"]
    trial_wf = p["trial_wf"]
    key = (method, trial_wf)
    key in SUPPORTED || error(
        "combination method=$method trial_wf=$trial_wf is not implemented yet")
    observables = parse_observables(p, method)
    println("Running method=$method trial_wf=$trial_wf")
    println("observables requested: ", isempty(observables) ? "none" : join(observables, ","))
    if method in ("vmc", "vmc_parallel")
        return run_vmc_2b_coupled(p)
    end
    return run_dmc_2b_coupled(p, method)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
