module Relief

using StatsBase
using Statistics
using LinearAlgebra
using Printf

export 
    relief,
    relieff,
    reliefseq,
    reliefmss,
    surfstar,
    surf,
    multisurfstar,
    multisurf,
    swrfstar,
    boostedsurf,
    iterative_relief,
    irelief,
    ecrelieff,
    vlsrelief,
    turf

# source files

include("./utils/square_to_vec.jl")

include("relief.jl")
include("relieff.jl")
include("reliefseq.jl")
include("reliefmss.jl")
include("surfstar.jl")
include("surf.jl")
include("multisurfstar.jl")
include("multisurf.jl")
include("swrfstar.jl")
include("boostedsurf.jl")
include("iterative_relief.jl")
include("irelief.jl")
include("ecrelieff.jl")
include("vlsrelief.jl")
include("turf.jl")

end
