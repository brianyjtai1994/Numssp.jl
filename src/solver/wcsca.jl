# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Water Cycle Sine-Cosine Algorithm Types Design
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
abstract type AbstractWCSCA{T} <: AbstractNumssp{T} end

# WCSCA System, stands for the whole population
struct WCSCAS{T} <: AbstractWCSCA{T}
    xpop::Matrix{T} # Current Position
    fval::Vector{T} # Current function-value
    viol::Vector{T} # Violation
    feas::BitVector # Feasibility
end

function WCSCAS(lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U) where {ND, T, U}
    xpop = Matrix{T}(undef, ND, NP)

    @inbounds for j in axes(xpop, 2)
        @simd for i in axes(xpop, 1)
            xpop[i,j] = lb[i] + rand(T) * (ub[i] - lb[i])
        end
    end

    fval = Vector{T}(undef, NP)
    viol = Vector{T}(undef, NP)
    feas = BitVector(undef, NP)

    return WCSCAS(xpop, fval, viol, feas)
end

# Elitist part of WCSCA System
struct Rivers{T, B, L, IM, IV} <: AbstractWCSCA{T}
    xpop::SubArray{T, 2, Matrix{T}, IM, L}
    fval::SubArray{T, 1, Vector{T}, IV, L}
    viol::SubArray{T, 1, Vector{T}, IV, L}
    feas::SubArray{B, 1, BitVector, IV, L}
end

# Other candidates of WCSCA System
struct Creeks{T, B, L, IM, IV} <: AbstractWCSCA{T}
    xpop::SubArray{T, 2, Matrix{T}, IM, L}
    fval::SubArray{T, 1, Vector{T}, IV, L}
    viol::SubArray{T, 1, Vector{T}, IV, L}
    feas::SubArray{B, 1, BitVector, IV, L}
end

Rivers(w::WCSCAS{T}, NR::U)        where {T, U} = Rivers(view(w.xpop, :,    1:NR), view(w.fval,    1:NR), view(w.viol,    1:NR), view(w.feas,    1:NR))
Creeks(w::WCSCAS{T}, NR::U, NP::U) where {T, U} = Creeks(view(w.xpop, :, NR+1:NP), view(w.fval, NR+1:NP), view(w.viol, NR+1:NP), view(w.feas, NR+1:NP))

include("constraint.jl")
include("benchmark_functions.jl"); export rosen!, ackley!, rastrigin!, gen_fitdata!
include("history.jl"); export evolve_benchmark

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Operation on Types of WCSCA
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
include("operand.jl")

_trial!(vecb::Vector{T}, rv::WR, cr::WC, node::U, jdx::U, ss::T)   where {T, U, WR, WC}  = _trial!(vecb, rv.xpop, cr.xpop, node, jdx, ss)
_trial!(vecb::Vector{T}, rv::WR, jdx::U, ss::T)                    where {T, U, WR}      = _trial!(vecb, rv.xpop, jdx, ss) # ss := step size

_rain!(vecb::Vector{T}, rv::WR, cr::WC, jdx::U, dmax::T)           where {T, U, WR, WC}  = _rain!(vecb, rv.xpop, cr.xpop, jdx, dmax)
_rain!(vecb::Vector{T}, rv::WR, lb::NDT, ub::NDT, jdx::U, dmax::T) where {T, U, WR, NDT} = _rain!(vecb, rv.xpop, lb, ub, jdx, dmax)

_init!(fn!::F, con::C, wca::W)                                     where {F, C, T, W}    = _init!(fn!, con, wca.xpop, wca.fval, wca.viol, wca.feas)
_sort!(wca::W, matb::Matrix{T})                                    where {T, W}          = _sort!(matb, wca.xpop, wca.fval, wca.viol, wca.feas)

_group!(fork::Vector{U}, wca::W, NR::U, NC::U)                     where {T, U, W}       = _group!(fork, wca.fval, NR, NC)
_match!(fn!::F, con::C, wca::W, xnew::Vector{T}, rdx::U, jdx::U)   where {F, C, W, T, U} = _match!(fn!, con, xnew, rdx, jdx, wca.xpop, wca.fval, wca.viol, wca.feas)
_renew!(wca::W)                                                    where W               = _renew!(wca.fval, wca.viol, wca.feas)

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Extended Functionality of WCSCA
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
include("curvefit.jl"); export decay_fit, guass_fit, lorentz_fit
