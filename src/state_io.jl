using Serialization

function save_dmc_state(path::AbstractString; state::NamedTuple)
    open(path, "w") do io
        serialize(io, state)
    end
    return path
end

load_dmc_state(path::AbstractString) = open(deserialize, path)
