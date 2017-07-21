# This file is a part of BAT.jl, licensed under the MIT License (MIT).

__precompile__(true)

module BAT

include.([
    "rand.jl",
    "util.jl",
    "onlinestats.jl",
    "parameters.jl",
    "targetfunction.jl",
    "proposalfunction.jl",
    "mhsampler.jl",
])

end # module
