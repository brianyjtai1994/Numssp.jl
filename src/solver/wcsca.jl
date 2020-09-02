#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Water Cycle Sine-Cosine Algorithm Types Design
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
abstract type AbstractWCSCA{T} <: AbstractNumssp{T} end

mutable struct Waters{T} <: AbstractWCSCA{T}
    x::Vector{T}; f::T; v::Bool; c::T
    #=
    x := parameters passed into models
    f := function-value of fn!(x)
    v := viability / feasibility
    c := contravention / violation
    =#

    function Waters(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND, T}
        x = Vector{T}(undef, ND)

        @inbounds @simd for i in eachindex(x)
            x[i] = lb[i] + rand(T) * (ub[i] - lb[i])
        end

        return new{T}(x, Inf, false, 0.0)
    end
end

function _waters(lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U) where {ND, T, U}
    wats = Vector{Waters{T}}(undef, NP) # 2 allocations for each Waters{T}

    @inbounds @simd for i in eachindex(wats)
        wats[i] = Waters(lb, ub)
    end

    return wats
end

function show(io::IO, w::Waters{T}) where T
    if abs(w.f) < 1000.0
        @printf "func-value: %.2f\n" w.f
    else
        @printf "func-value: %.2e\n" w.f
    end

    @printf "  feasible: %s\n" w.v
    @printf " violation: %.2f\n" w.c

    print("parameters:\n [")

    @inbounds for i in 1:length(w.x)-1
        val = w.x[i]
        if abs(val) < 1000.0
            @printf "%.2f, " val
        else
            @printf "%.2e, " val
        end
    end

    if abs(w.x[end]) < 1000.0
        @printf "%.2f]\n" w.x[end]
    else
        @printf "%.2e]\n" w.x[end]
    end
end

function ==(w1::Waters{T}, w2::Waters{T}) where T
    # w1, w2 are both feasible
    if w1.v && w2.v
        return w1.f == w2.f

    # w1, w2 are both infesasible
    elseif !w1.v && !w2.v
        return w1.c == w2.c

    else
        return false
    end
end

function isless(w1::Waters{T}, w2::Waters{T}) where T
    # w1, w2 are both feasible
    if w1.v && w2.v
        return w1.f < w2.f

    # w1, w2 are both infesasible
    elseif !w1.v && !w2.v
        return w1.c < w2.c

    # if (w1, w2) = (feasible, infeasible), then w1 < w2
    # if (w1, w2) = (infeasible, feasible), then w2 < w1
    else
        return w1.v
    end
end

_rivers(wats::Vector{Waters{T}}, NR::Int)          where T = view(wats, 1:NR)
_creeks(wats::Vector{Waters{T}}, NR::Int, NP::Int) where T = view(wats, NR+1:NP)

struct WCSCALog{T, U} <: AbstractWCSCA{T}
    xsol::Matrix{T}    # The best position, size = (ND, imax)
    fsol::Vector{T}    # The best solution, size = (imax,)
    fork::Matrix{U}    # Histories of fork, size = (NR, imax)
    dims::NTuple{3, U}

    function WCSCALog(ND::U, NR::U, imax::U, ::Val{T}) where {T<:Number, U}
        return new{T, U}(Matrix{T}(undef, ND, imax), Vector{T}(undef, imax),
                         Matrix{U}(undef, NR, imax), (ND, NR, imax))
    end
end

function _update!(logs::WCSCALog{T,U}, sea::Waters{T}, forks::Vector{U}, it::U) where {T, U}
    @inbounds @simd for i in eachindex(sea.x)
        logs.xsol[i, it] = sea.x[i]
    end

    logs.fsol[it] = sea.f

    @inbounds @simd for i in eachindex(forks)
        logs.fork[i, it] = forks[i]
    end
end

include("constraint.jl")
include("operand.jl")
include("curvefit.jl")
include("benchmarkf.jl")
