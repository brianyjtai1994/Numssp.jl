export decay_fit, guass_fit, lorentz_fit

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Simple/Weighted Least Sqaure Type Definition/Operation
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
abstract type AbstractFit{T} <: Function       end
abstract type AbstractLSQ{T} <: AbstractFit{T} end
abstract type AbstractχSQ{T} <: AbstractFit{T} end

# @code_warntype ✓
function _lsq!(x::VT, y::VT, p::VT, f::F) where {F, T, VT<:Vector{T}}
    ret = 0.0

    @inbounds for i in eachindex(x)
        ret += (y[i] - func(x[i], p))^2.0
    end

    return ret
end

# @code_warntype ✓
function _χsq!(x::VT, y::VT, σ::VT, p::VT, f::F) where {F, T, VT<:Vector{T}}
    ret = 0.0

    @inbounds for i in eachindex(x)
        ret += σ[i] * (y[i] - f(x[i], p))^2.0
    end

    return ret
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Multi-Exponential Decay LeastSQ
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
struct DecayLSQ{T} <: AbstractLSQ{T} x::Vector{T}; y::Vector{T}               end
struct DecayχSQ{T} <: AbstractχSQ{T} x::Vector{T}; y::Vector{T}; σ::Vector{T} end

function decay_fit(x::VT, y::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND & 1 ≠ 1
        return error("Invalid boundary dimension (odd number for multi-exponential decay).")
    end

    return curve_fit(DecayLSQ(x, y), lb, ub)
end

function decay_fit(x::VT, y::VT, σ::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND & 1 ≠ 1
        return error("Invalid boundary dimension (odd number for multi-exponential decay).")
    end

    return curve_fit(DecayχSQ(x, y, σ), lb, ub)
end

(lsq::DecayLSQ{T})(p::Vector{T}) where T = _lsq!(lsq.x, lsq.y, p, _decay)
(χsq::DecayχSQ{T})(p::Vector{T}) where T = _χsq!(χsq.x, χsq.y, χsq.σ, p, _decay)

function _decay(x::T, p::Vector{T}) where T
    ret = p[end]

    @inbounds for i in 1:div(length(p), 2)
        ret += p[2i-1] * exp(- x / p[2i])
    end

    return ret
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Gaussian LeastSQ
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
struct GaussLSQ{T} <: AbstractLSQ{T} x::Vector{T}; y::Vector{T}               end
struct GaussχSQ{T} <: AbstractχSQ{T} x::Vector{T}; y::Vector{T}; σ::Vector{T} end

function gauss_fit(x::VT, y::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Gaussian distribution).")
    end

    return curve_fit(GaussLSQ(x, y), lb, ub)
end

function gauss_fit(x::VT, y::VT, σ::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Gaussian distribution).")
    end

    return curve_fit(GaussχSQ(x, y, σ), lb, ub)
end

(lsq::GaussLSQ{T})(p::Vector{T}) where T = _lsq!(lsq.x, lsq.y, p, _gauss)
(χsq::GaussχSQ{T})(p::Vector{T}) where T = _χsq!(χsq.x, χsq.y, χsq.σ, p, _gauss)

_gauss(x::T, p::Vector{T})           where T = _gauss(x, p[1], p[2], p[3], p[4])
_gauss(x::T, A::T, μ::T, σ::T, c::T) where T = (A / (σ * sqrt(2.0π))) * exp(-0.5 * ((x - μ) / σ)^2.0) + c

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Lorentzian LeastSQ
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
struct LorentzLSQ{T} <: AbstractLSQ{T} x::Vector{T}; y::Vector{T}                  end
struct LorentzχSQ{T} <: AbstractχSQ{T} x::Vector{T}; y::Vector{T}; σ::Vector{T} end

function lorentz_fit(x::VT, y::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Lorentzian distribution).")
    end

    return curve_fit(LorentzLSQ(x, y), lb, ub)
end

function lorentz_fit(x::VT, y::VT, σ::VT, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, VT<:Vector{T}}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Lorentzian distribution).")
    end

    return curve_fit(LorentzχSQ(x, y, σ), lb, ub)
end

(lsq::LorentzLSQ{T})(p::Vector{T}) where T = _lsq!(lsq.x, lsq.y, p, _lorentz)
(χsq::LorentzχSQ{T})(p::Vector{T}) where T = _χsq!(χsq.x, χsq.y, χsq.σ, p, _lorentz)

_lorentz(x::T, p::Vector{T})           where T = _lorentz(x, p[1], p[2], p[3], p[4])
_lorentz(x::T, A::T, μ::T, Γ::T, c::T) where T = (A / π) * (0.5Γ / ((x - μ)^2.0 + (0.5Γ)^2.0)) + c

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 WCSCA Curve Fitting
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
function curve_fit(sq::SQ, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND, SQ<:AbstractFit{T}}
    NP   = 30 * ND
    NR   = ND + 1
    imax = 180 * ND

    minimize!(sq, lb, ub, NP, NR, imax, 1e-7)
end
