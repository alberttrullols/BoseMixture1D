# Machine-specific SLURM and module settings.
# CPUs per task and time limit are run settings and live in generate_folders.jl.
const CORREFOC = (
    partition = "high-cpu",
    account   = "lab_upc59",
    qos       = "res_upc59_a",
    modules   = ("module load modulepath/EESSI/2025.06", "module load Julia/1.12.2"),
)

const MN5 = (
    partition = "",
    account   = "upc59",
    qos       = "gp_resc",
    modules   = ("module load julia",),
)

const MACHINE = CORREFOC
