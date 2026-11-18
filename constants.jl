# String values are written into folder names and params.dat, so they must match the existing run folders.

const VMC           = "vmc"
const VMC_PARALLEL  = "vmc_parallel"
const DMC_MIXED     = "dmc_mixed"
const DMC_PURE      = "dmc_pure"
const DMC_PURE_OVLP = "dmc_pure_ovlp"

const TRIAL_2B_UNCOUPLED       = "2b_uncoupled"
const TRIAL_2B_UNCOUPLED_MATCHING = "2b_uncoupled_matching"
const TRIAL_2B_PHONONS         = "2b_phonons"

const METHODS   = (VMC, VMC_PARALLEL, DMC_MIXED, DMC_PURE, DMC_PURE_OVLP)
const TRIAL_WFS = (TRIAL_2B_UNCOUPLED, TRIAL_2B_UNCOUPLED_MATCHING, TRIAL_2B_PHONONS)

const OBS_CS_R      = :Cs_r
const OBS_CN_R      = :Cn_r
const OBS_SK        = :Sk
const OBS_SN        = :Sn

const VMC_OBSERVABLES = Set([OBS_CS_R, OBS_CN_R, OBS_SK, OBS_SN])
const DMC_OBSERVABLES = Set([OBS_CS_R, OBS_SK, OBS_SN])

const SUPPORTED_OBSERVABLES = Dict(
    VMC           => VMC_OBSERVABLES,
    VMC_PARALLEL  => VMC_OBSERVABLES,
    DMC_MIXED     => DMC_OBSERVABLES,
    DMC_PURE      => DMC_OBSERVABLES,
    DMC_PURE_OVLP => DMC_OBSERVABLES,
)
