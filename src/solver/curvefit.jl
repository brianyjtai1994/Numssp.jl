# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Simple/Weighted Least Sqaure Type Definition/Operation
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
abstract type AbstractSQ{T}  <: Function      end
abstract type AbstractLSQ{T} <: AbstractSQ{T} end
abstract type AbstractχSQ{T} <: AbstractSQ{T} end

# @code_warntype ✓
function _lsq!(xdat::Vector{T}, ydat::Vector{T}, params::VT, func::F) where {F, T, VT<:AbstractVector{T}}
    ret = 0.0

    @inbounds for i in eachindex(xdat)
        ret += (ydat[i] - func(xdat[i], params))^2.0
    end

    return ret
end

# @code_warntype ✓
function _lsq!(xdat::Vector{T}, ydat::Vector{T}, xm::MT, jdx::U, func::F) where {F, T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in eachindex(xdat)
        ret += (ydat[i] - func(xdat[i], xm, jdx))^2.0
    end

    return ret
end

# @code_warntype ✓
function _lsq!(xdat::Vector{T}, ydat::Vector{T}, des::Vector{T}, xm::MT, func::F) where {F, T, MT<:AbstractMatrix{T}}
    @inbounds for jdx in eachindex(des)
        des[jdx] = _lsq!(xdat, ydat, xm, jdx, func)
    end
end

# @code_warntype ✓
function _χsq!(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, params::VT, func::F) where {F, T, VT<:AbstractVector{T}}
    ret = 0.0

    @inbounds for i in eachindex(xdat)
        ret += σdat[i] * (ydat[i] - func(xdat[i], params))^2.0
    end

    return ret
end

# @code_warntype ✓
function _χsq!(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, xm::MT, jdx::U, func::F) where {F, T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in eachindex(xdat)
        ret += σdat[i] * (ydat[i] - func(xdat[i], xm, jdx))^2.0
    end

    return ret
end

# @code_warntype ✓
function _χsq!(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, des::Vector{T}, xm::MT, func::F) where {F, T, MT<:AbstractMatrix{T}}
    @inbounds for jdx in eachindex(des)
        des[jdx] = _χsq!(xdat, ydat, σdat, xm, jdx, func)
    end
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Multi-Exponential Decay LeastSQ
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
struct DecayLSQ{T} <: AbstractLSQ{T} xdat::Vector{T}; ydat::Vector{T}                  end
struct DecayχSQ{T} <: AbstractχSQ{T} xdat::Vector{T}; ydat::Vector{T}; σdat::Vector{T} end

decay_fit(xdat::Vector{T}, ydat::Vector{T},                  lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND} = curve_fit(DecayLSQ(xdat, ydat),       lb, ub)
decay_fit(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND} = curve_fit(DecayχSQ(xdat, ydat, σdat), lb, ub)

(lsq::DecayLSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _lsq!(lsq.xdat, lsq.ydat,  params, _decay)
(lsq::DecayLSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _lsq!(lsq.xdat, lsq.ydat, xm, jdx, _decay)
(lsq::DecayLSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _lsq!(lsq.xdat, lsq.ydat, des, xm, _decay)

(χsq::DecayχSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat,  params, _decay)
(χsq::DecayχSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, xm, jdx, _decay)
(χsq::DecayχSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, des, xm, _decay)

function _decay(xdat::T, params::VT) where {T, VT<:AbstractVector{T}}
    ret = params[end]

    @inbounds for i in 1:length(params)÷2
        ret += params[2i-1] * exp(- xdat / params[2i])
    end

    return ret
end

function _decay(xdat::T, xm::MT, jdx::U) where {T, U, MT<:AbstractMatrix{T}}
    ret = xm[end, jdx]

    @inbounds for i in 1:size(xm,1)÷2
        ret += xm[2i-1, jdx] * exp(- xdat / xm[2i, jdx])
    end

    return ret
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Gaussian LeastSQ
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
struct GaussLSQ{T} <: AbstractLSQ{T} xdat::Vector{T}; ydat::Vector{T}                  end
struct GaussχSQ{T} <: AbstractχSQ{T} xdat::Vector{T}; ydat::Vector{T}; σdat::Vector{T} end

function gauss_fit(xdat::Vector{T}, ydat::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Gaussian distribution).")
    end

    return curve_fit(GaussLSQ(xdat, ydat), lb, ub)
end

function gauss_fit(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Gaussian distribution).")
    end

    return curve_fit(GaussχSQ(xdat, ydat, σdat), lb, ub)
end

(lsq::GaussLSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _lsq!(lsq.xdat, lsq.ydat,  params, _gauss)
(lsq::GaussLSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _lsq!(lsq.xdat, lsq.ydat, xm, jdx, _gauss)
(lsq::GaussLSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _lsq!(lsq.xdat, lsq.ydat, des, xm, _gauss)

(χsq::GaussχSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat,  params, _gauss)
(χsq::GaussχSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, xm, jdx, _gauss)
(χsq::GaussχSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, des, xm, _gauss)

_gauss(xdat::T, xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _gauss(xdat, xm[1, jdx], xm[2, jdx], xm[3, jdx], xm[4, jdx])
_gauss(xdat::T, params::VT)             where {T, VT<:AbstractVector{T}}    = _gauss(xdat, params[1], params[2], params[3], params[4])
_gauss(xdat::T, A::T, μ::T, σ::T, c::T) where T                             = (A / (σ * sqrt(2.0π))) * exp(-0.5 * ((x - μ) / σ)^2.0) + c

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Lorentzian LeastSQ
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
struct LorentzLSQ{T} <: AbstractLSQ{T} xdat::Vector{T}; ydat::Vector{T}                  end
struct LorentzχSQ{T} <: AbstractχSQ{T} xdat::Vector{T}; ydat::Vector{T}; σdat::Vector{T} end

function lorentz_fit(xdat::Vector{T}, ydat::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Lorentzian distribution).")
    end

    return curve_fit(LorentzLSQ(xdat, ydat), lb, ub)
end

function lorentz_fit(xdat::Vector{T}, ydat::Vector{T}, σdat::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {T, ND}
    if ND ≠ 4
        return error("Invalid boundary dimension (4 for Lorentzian distribution).")
    end

    return curve_fit(LorentzχSQ(xdat, ydat, σdat), lb, ub)
end

(lsq::LorentzLSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _lsq!(lsq.xdat, lsq.ydat,  params, _lorentz)
(lsq::LorentzLSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _lsq!(lsq.xdat, lsq.ydat, xm, jdx, _lorentz)
(lsq::LorentzLSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _lsq!(lsq.xdat, lsq.ydat, des, xm, _lorentz)

(χsq::LorentzχSQ{T})(params::VT)             where {T, VT<:AbstractVector{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat,  params, _lorentz)
(χsq::LorentzχSQ{T})(xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, xm, jdx, _lorentz)
(χsq::LorentzχSQ{T})(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}    = _χsq!(χsq.xdat, χsq.ydat, χsq.σdat, des, xm, _lorentz)

_lorentz(xdat::T, xm::MT, jdx::U)         where {T, U, MT<:AbstractMatrix{T}} = _lorentz(xdat, xm[1, jdx], xm[2, jdx], xm[3, jdx], xm[4, jdx])
_lorentz(xdat::T, params::VT)             where {T, VT<:AbstractVector{T}}    = _lorentz(xdat, params[1], params[2], params[3], params[4])
_lorentz(xdat::T, A::T, μ::T, Γ::T, c::T) where T                             = (A / π) * (0.5Γ / ((x - μ)^2.0 + (0.5Γ)^2.0)) + c

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# WCSCA Curve Fitting
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
function curve_fit(sq::SQ, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U=25, NR::U=2, dmax::T=1e-7) where {SQ<:AbstractSQ, T, U, ND}
    NP = NP ≥ 25 * ND ? NP : 25 * ND
    NR = NR > ND ? NR : NR = ND + 1
    NC = NP - NR

    it_max = 300.0 * ND; it = 0.0

    wcscas = WCSCAS(lb, ub, NP)
    rivers = Rivers(wcscas, NR)
    creeks = Creeks(wcscas, NR, NP)

    constraint = Constraints(lb, ub)

    mbuff = Matrix{T}(undef, ND, 2)
    vbuff = Vector{T}(undef, ND)
    forks = Vector{U}(undef, NR)

    _init!(sq, constraint, wcscas)
    _sort!(wcscas, mbuff)

    while it < it_max
        it += 1.0
        ss  = 2.0 - _sigmoid(it, 0.5 * it_max, 0.618, 20.0 / it_max)

        _evolve!(sq, constraint, wcscas, rivers, creeks, vbuff, forks, ss, NR, NC)
        _evapor!(sq, constraint, wcscas, rivers, creeks, vbuff, forks, lb, ub, dmax, NR)

        _renew!(wcscas); _sort!(wcscas, mbuff)

        dmax -= dmax / it_max
    end

    return wcscas.xpop[:, 1], wcscas.fval[1]
end
