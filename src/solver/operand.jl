# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Fundamental Operations
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

_sigmoid(x::T, x₀::T, a::T, k::T, offset::T) where T = a / (1.0 + exp(k * (x₀ - x))) + offset
_sigmoid(x::T, x₀::T, a::T, k::T)            where T = _sigmoid(x, x₀, a, k, 0.0)

# @code_warntype ✓
function _trial!(vecb::Vector{T}, rv::MT, cr::MT, rdx::U, jdx::U, ss::T) where {T, U, MT<:AbstractMatrix{T}}
    r = rand(T) * 2.0

    @inbounds @simd for i in eachindex(vecb)
        vecb[i] = cr[i,jdx] + ss * abs(rv[i,rdx] - cr[i,jdx]) * ifelse(rand(T) < 0.5, sinpi(r), cospi(r)) # SCA
    end
end

# @code_warntype ✓
function _trial!(vecb::Vector{T}, rv::MT, jdx::U, ss::T) where {T, U, MT<:AbstractMatrix{T}}
    r = rand(T) * 2.0

    @inbounds @simd for i in eachindex(vecb)
        vecb[i] = rv[i,jdx] + ss * abs(rv[i,1] - rv[i,jdx]) * ifelse(rand(T) < 0.5, sinpi(r), cospi(r)) # SCA
    end
end

# @code_warntype ✓
function _rain!(vecb::Vector{T}, rv::MT) where {T, MT<:AbstractMatrix{T}}
    r = randn(T)

    @inbounds for i in eachindex(vecb)
        vecb[i] = rv[i,1] + r * sqrt(0.1)
    end
end

# @code_warntype ✓
function _rain!(vecb::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND, T}
    r = rand(T)

    @inbounds for i in eachindex(vecb)
        vecb[i] = lb[i] + r * (ub[i] - lb[i])
    end
end

# @code_warntype ✓
function _dist(rv::MT, jrv::U, cr::MT, jcr::U) where {T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in axes(rv, 1)
        ret += (rv[i, jrv] - cr[i, jcr])^2.0
    end

    return sqrt(ret)
end

# @code_warntype ✓
function _dist(rv::MT, jdx1::U, jdx2::U) where {T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in axes(rv, 1)
        ret += (rv[i, jdx1] - rv[i, jdx2])^2.0
    end

    return sqrt(ret)
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Core Operations of WCSCA
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# initialize the population stored in WCSCAS, @code_warntype ✓
function _init!(fn!::F, cons::C, xpop::Matrix{T}, fval::Vector{T}, viol::Vector{T}, feas::BitVector) where {F, C, T}
    fmax = -Inf

    @inbounds for j in eachindex(fval)
        violation = _constraint(cons, xpop, j)

        if violation > 0.0
            viol[j] = violation
        else
            feas[j] = true
            fval[j] = fn!(xpop, j)
            fmax    = max(fmax, fval[j])
        end
    end

    @inbounds @simd for j in eachindex(fval)
        if !feas[j]
            fval[j] = viol[j] + fmax
        end
    end
end

# parallel insertion-sort of WCSCAS according to the function value, @code_warntype ✓
function _sort!(matb::Matrix{T}, xpop::Matrix{T}, fval::Vector{T}, viol::Vector{T}, feas::BitVector) where T
    ND, NP = size(xpop)

    @inbounds for j in 2:NP
        @simd for i in 1:ND
            matb[i, 1] = xpop[i, j]
        end

        fvalj = fval[j]
        violj = viol[j]
        feasj = feas[j]

        k = j - 1

        while k > 0 && fval[k] > fvalj
            @simd for i in 1:ND
                xpop[i, k + 1] = xpop[i, k]
            end

            fval[k + 1] = fval[k]
            viol[k + 1] = viol[k]
            feas[k + 1] = feas[k]

            k -= 1
        end

        @simd for i in 1:ND
            xpop[i, k + 1] = matb[i, 1]
        end

        fval[k + 1] = fvalj
        viol[k + 1] = violj
        feas[k + 1] = feasj
    end
end

# assign the number of Creeks flowing into Rivers, @code_warntype ✓
function _group!(fork::Vector{U}, fval::Vector{T}, NR::U, NC::U) where {T, U}
    diversity = 0.0

    @inbounds for i in eachindex(fork)
        diversity += fval[NR + 1] - fval[i]
    end

    if iszero(diversity) || isnan(diversity)
        fill!(fork, 1)
    else
        @inbounds @simd for i in eachindex(fork)
            fork[i] = max(1, round(Int, NC * (fval[NR + 1] - fval[i]) / diversity))
        end
    end

    residue = NC; idx = 2

    @inbounds for i in eachindex(fork)
        residue -= fork[i]
    end

    while residue > 0
        fork[idx] += 1
        residue   -= 1

        idx < NR ? idx += 1 : idx = 2
    end

    while residue < 0
        fork[idx] = max(1, fork[idx] - 1)
        residue  += 1

        idx < NR ? idx += 1 : idx = 2
    end
end

# @code_warntype ✓
function _rain!(vecb::Vector{T}, rv::MT, cr::MT, jdx::U, dmax::T) where {T, U, MT<:AbstractMatrix{T}}
    if _dist(rv, 1, cr, jdx) > dmax
        return false

    else
        _rain!(vecb, rv)

        return true
    end
end

# @code_warntype ✓
function _rain!(vecb::Vector{T}, rv::MT, lb::NTuple{ND,T}, ub::NTuple{ND,T}, jdx::U, dmax::T) where {ND, T, U, MT<:AbstractMatrix{T}}
    if _dist(rv, 1, jdx) > dmax && rand(T) > 0.1
        return false

    else
        _rain!(vecb, lb, ub)

        return true
    end
end

# tournament with feasibility handling, @code_warntype ✓
function _match!(fn!::F, cons::C, xnew::Vector{T}, rdx::U, jdx::U, xpop::MT, fval::VT, viol::VT, feas::VB) where {F, C, B, T, U, VT<:AbstractVector{T}, VB<:AbstractVector{B}, MT<:AbstractMatrix{T}}
    violation = _constraint(cons, xnew)

    # x[new] is infeasible
    if violation > 0.0
        _match!(xnew, violation, jdx, xpop, viol, feas)

    # x[new] is feasible
    else
        fnew = fn!(xnew)

        _match!(xnew, fnew, rdx, jdx, xpop, fval, viol, feas)
    end
end

# Matchup for a feasible new-trial, @code_warntype ✓
function _match!(xnew::Vector{T}, fnew::T, rdx::U, jdx::U, xpop::MT, fval::VT, viol::VT, feas::VB) where {B, T, U, VT<:AbstractVector{T}, VB<:AbstractVector{B}, MT<:AbstractMatrix{T}}
    # x[old] is infeasible
    if !feas[jdx]
        fval[jdx] = fnew
        feas[jdx] = true
        viol[jdx] = 0.0

        @inbounds @simd for i in eachindex(xnew)
            xpop[i, jdx] = xnew[i]
        end

    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    elseif fnew ≤ fval[rdx]
        @inbounds @simd for i in eachindex(xnew)
            temp = xpop[i, rdx]; xpop[i, rdx] = xnew[i]; xpop[i, jdx] = temp
        end

        temp = fval[rdx]; fval[rdx] = fnew; fval[jdx] = temp

    # x[old], x[new] are feasible
    elseif fnew ≤ fval[jdx]
        fval[jdx] = fnew

        @inbounds @simd for i in eachindex(xnew)
            xpop[i, jdx] = xnew[i]
        end
    end
end

# Matchup for an infeasible new trial, here `fnew == violation`, @code_warntype ✓
function _match!(xnew::Vector{T}, fnew::T, jdx::U, xpop::MT, viol::VT, feas::VB) where {B, T, U, VT<:AbstractVector{T}, VB<:AbstractVector{B}, MT<:AbstractMatrix{T}}
    # x[old], x[new] are infeasible, compare violation
    if !feas[jdx] && fnew ≤ viol[jdx]
        viol[jdx] = fnew

        @inbounds @simd for i in eachindex(xnew)
            xpop[i, jdx] = xnew[i]
        end
    end
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# WCSCA Operations of Each Generation of Evolution
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =

# update the function-value of infeasible candidates, @code_warntype ✓
function _renew!(fval::Vector{T}, viol::Vector{T}, feas::BitVector) where T
    fmax = -Inf

    @inbounds for j in eachindex(fval)
        # xj is feasible
        if feas[j]
            fmax = max(fmax, fval[j])
        end
    end

    @inbounds @simd for j in eachindex(fval)
        if !feas[j]
            fval[j] = viol[j] + fmax
        end
    end
end

# @code_warntype ✓
function _evolve!(fn!::F, cons::C, wca::W, river::WR, creek::WC, vecb::Vector{T}, fork::Vector{U}, ss::T, NR::U, NC::U) where {F, C, T, U, W, WR, WC}
    _group!(fork, wca, NR, NC)

    river_idx   = 1
    fork_res, s = iterate(fork)

    # flowing creeks to rivers
    @inbounds for jdx in 1:NC
        _trial!(vecb, river, creek, river_idx, jdx, ss)
        _match!(fn!, cons, wca, vecb, river_idx, NR + jdx)

        fork_res -= 1

        if iszero(fork_res)
            river_idx += 1; y = iterate(fork, s)

            if y ≠ nothing
                fork_res, s = y
            end
        end
    end

    # flowing rivers to sea
    @inbounds for jdx in 2:NR
        _trial!(vecb, river, jdx, ss)
        _match!(fn!, cons, river, vecb, 1, jdx)
    end
end

# @code_warntype ✓
function _evapor!(fn!::F, cons::C, wca::W, river::WR, creek::WC, vecb::Vector{T}, fork::Vector{U}, lb::NDT, ub::NDT, dmax::T, NR::U) where {F, C, T, U, W, WR, WC, NDT}
    @inbounds for jdx in 1:fork[1]
        if_rain = _rain!(vecb, river, creek, jdx, dmax)

        if if_rain
            _match!(fn!, cons, wca, vecb, 1, NR + jdx)
        end
    end

    @inbounds for jdx in 2:NR
        if_rain = _rain!(vecb, river, lb, ub, jdx, dmax)

        if if_rain
            _match!(fn!, cons, river, vecb, 1, jdx)
        end
    end
end
