include(joinpath(@__DIR__, "generate_folders.jl"))

const METHOD   = DMC_MIXED
const TRIAL_WF = TRIAL_2B_UNCOUPLED
const PREFIX   = "N80_mixture"

const OBSERVABLES = Dict{Symbol, Bool}(
    OBS_CS_R => true,
    OBS_CN_R => false,
    OBS_SK   => true,
    OBS_SN   => true,
)


N = 80 # total number of particles
g = 0.01
dt = 0.025
size_pairs = [(Int(N/2), Int(N/2))]


# dmc run parameters
num_steps = 1_000_000
num_walkers = 1000
time_order = 2
measurement_stride = 1_000
equilibration_steps = 150_000

# vmc run parameters
vmc_equilibration_steps = 250_000
vmc_production_steps = 6_000_000
vmc_step_size = 7.5     # VMC Metropolis step size


save_state = false      # DMC only, write final walker state to final_state.jls at end of run

# dmc pure parameters
acum_mixed = true # accumulate mixed estimators while running DMC pure?
forward_walk_steps_values = [30_000] 


# uncoupled matching trial wf parameters
R_match_same_values = Float64[] 
R_match_ud_values = Float64[] 


# inter-species
# g12_values = [0.01258, 0.01272, 0.012860, 0.013]
g12_values = [0.012694]

parent_folder = PREFIX

total = generate_folders(;
    g, dt, size_pairs, num_steps, num_walkers,
    forward_walk_steps_values, time_order, g12_values,
    method=METHOD, trial_wf=TRIAL_WF, observables=OBSERVABLES, prefix=PREFIX,
    parent_folder, acum_mixed, measurement_stride,
    equilibration_steps, vmc_equilibration_steps, vmc_production_steps, vmc_step_size,
    save_state, R_match_same_values, R_match_ud_values)

println("Total folders created: $total")
