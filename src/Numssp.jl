module Numssp

using Printf

abstract type AbstractNumssp{T} end

include("fft.jl")
include("solver/wcsca.jl")

end # module
