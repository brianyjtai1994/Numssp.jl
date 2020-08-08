# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# History of WCSCA
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
struct WCSCALog{T, U} <: AbstractWCSCA{T}
    log_xsol::Matrix{T} # The best position, size = (ND, itmax)
    log_fsol::Vector{T} # The best solution, size = (itmax,)
    log_fork::Matrix{U} # Histories of fork, size = (NR, itmax)
    log_dims::NTuple{3, U}
end

WCSCALog(ND::U, NR::U, itmax::T) where {T, U} =
    WCSCALog(Matrix{T}(undef, ND, U(itmax)), Vector{T}(undef, U(itmax)), Matrix{U}(undef, NR, U(itmax)), (ND, NR, U(itmax)))

_update!(wh::WCSCALog{T,U}, wca::WCSCAS{T}, fork::Vector{U}, it::U) where {T, U} =
    _update!(wh.log_xsol, wh.log_fsol, wh.log_fork, wca.xpop, wca.fval, fork, it)

function _update!(log_xsol::Matrix{T}, log_fsol::Vector{T}, log_fork::Matrix{U}, xpop::Matrix{T}, fval::Vector{T}, fork::Vector{U}, it::U) where {T, U}
    @inbounds @simd for i in axes(xpop, 1)
        log_xsol[i, it] = xpop[i,1]
    end

    log_fsol[it] = fval[1]

    @inbounds @simd for i in eachindex(fork)
        log_fork[i, it] = fork[i]
    end
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# WCSCA Evolution Benchmark
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
function evolve_benchmark(fn!::F, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U=25, NR::U=2, dmax::T=1e-7) where {F, T, U, ND}
    NP = NP ≥ 25 * ND ? NP : 25 * ND
    NR = NR > ND ? NR : NR = ND + 1
    NC = NP - NR

    it_max = 150.0 * ND; it = 0.0

    wcscas = WCSCAS(lb, ub, NP)
    rivers = Rivers(wcscas, NR)
    creeks = Creeks(wcscas, NR, NP)

    wcalog = WCSCALog(ND, NR, it_max)

    constraint = Constraints(lb, ub)

    mbuff = Matrix{T}(undef, ND, 2)
    vbuff = Vector{T}(undef, ND)
    forks = Vector{U}(undef, NR)

    _init!(fn!, constraint, wcscas)
    _sort!(wcscas, mbuff)

    while it < it_max
        it += 1.0
        ss  = 2.0 - _sigmoid(it, 0.5 * it_max, 0.618, 20.0 / it_max)

        _evolve!(fn!, constraint, wcscas, rivers, creeks, vbuff, forks, ss, NR, NC)
        _evapor!(fn!, constraint, wcscas, rivers, creeks, vbuff, forks, lb, ub, dmax, NR)

        _renew!(wcscas); _sort!(wcscas, mbuff)

        _update!(wcalog, wcscas, forks, U(it))

        dmax -= dmax / it_max
    end

    return wcalog
end
