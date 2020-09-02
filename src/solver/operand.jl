export minimize!

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Fundamental Operations
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

_sigmoid(x::T, x₀::T, a::T, k::T, offset::T) where T = a / (1.0 + exp(k * (x₀ - x))) + offset
_sigmoid(x::T, x₀::T, a::T, k::T)            where T = _sigmoid(x, x₀, a, k, 0.0)

# @code_warntype ✓
function _trial!(buff::Vector{T}, wat1::Vector{T}, wat2::Vector{T}, ss::T) where T
    r = rand(T) * 2.0

    @inbounds @simd for i in eachindex(buff)
        buff[i] = wat1[i] + ss * abs(wat1[i] - wat2[i]) * ifelse(rand(T) < 0.5, sinpi(r), cospi(r)) # SCA
    end
end

# @code_warntype ✓
function _rain!(buff::Vector{T}, sea::Vector{T}) where T
    r = randn(T)

    @inbounds for i in eachindex(buff)
        buff[i] = sea[i] + r * sqrt(0.1)
    end
end

# @code_warntype ✓
function _rain!(buff::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND, T}
    r = rand(T)

    @inbounds for i in eachindex(buff)
        buff[i] = lb[i] + r * (ub[i] - lb[i])
    end
end

# @code_warntype ✓
function _dist(v1::Vector{T}, v2::Vector{T}) where T
    ret = 0.0

    @inbounds for i in eachindex(v1)
        ret += (v1[i] - v2[i])^2.0
    end

    return sqrt(ret)
end

_sort!(wats::Vector{Waters{T}}) where T = binary_insertsort!(wats, 1, length(wats))

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Core Operations of WCSCA
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

# initialize the population stored in WCSCAS, @code_warntype ✓
function _init!(fn!::F, cons::C, wats::Vector{Waters{T}}) where {F, C, T}
    fmax = -Inf

    @inbounds for wat in wats
        violation = _constraint(cons, wat.x)

        # wat is infeasible
        if violation > 0.0
            wat.c = violation

        else
            wat.v = true
            wat.f = fn!(wat.x)
            fmax  = max(fmax, wat.f)
        end
    end

    @inbounds @simd for wat in wats
        # wat is infeasible
        if !wat.v
            wat.f = wat.c + fmax
        end
    end
end

# assign the number of Creeks flowing into Rivers, @code_warntype ✓
function _group!(fork::Vector{Int}, wats::Vector{Waters{T}}, NR::Int, NC::Int) where T
    diversity = 0.0

    @inbounds for i in eachindex(fork)
        diversity += wats[NR + 1].f - wats[i].f
    end

    if iszero(diversity) || isnan(diversity)
        fill!(fork, 1)
    else
        @inbounds @simd for i in eachindex(fork)
            fork[i] = max(1, round(Int, NC * (wats[NR + 1].f - wats[i].f) / diversity))
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
function _rain!(buff::Vector{T}, sea::Vector{T}, crk::Vector{T}, dmax::T) where T
    if _dist(sea, crk) > dmax
        return false

    else
        _rain!(buff, sea)

        return true
    end
end

# @code_warntype ✓
function _rain!(buff::Vector{T}, sea::Vector{T}, riv::Vector{T}, lb::NTuple{ND,T}, ub::NTuple{ND,T}, dmax::T) where {ND, T}
    if _dist(sea, riv) > dmax && rand(T) > 0.1
        return false

    else
        _rain!(buff, lb, ub)

        return true
    end
end

# tournament with feasibility handling of "creeks", @code_warntype ✓
function _match!(xnew::Vector{T}, wats::Vector{Waters{T}}, rivs::SubArray{Waters{T}}, crks::SubArray{Waters{T}}, rdx::Int, jdx::Int, fn!::F, cons::C) where {F, C, T}
    violation = _constraint(cons, xnew)

    # x[new] is infeasible
    if violation > 0.0
        _match!(xnew, violation, crks[jdx])

    # x[new] is feasible
    else
        fnew = fn!(xnew)

        _match!(xnew, fnew, wats, rivs, crks, rdx, jdx)
    end
end

# tournament with feasibility handling of "rivers", @code_warntype ✓
function _match!(xnew::Vector{T}, rivs::SubArray{Waters{T}}, rdx::Int, fn!::F, cons::C) where {F, C, T}
    violation = _constraint(cons, xnew)

    # x[new] is infeasible
    if violation > 0.0
        _match!(xnew, violation, rivs[rdx])

    # x[new] is feasible
    else
        fnew = fn!(xnew)

        _match!(xnew, fnew, rivs, rdx)
    end
end

# Matchup for a feasible x[new] trial in "creeks", @code_warntype ✓
function _match!(xnew::Vector{T}, fnew::T, wats::Vector{Waters{T}}, rivs::SubArray{Waters{T}}, crks::SubArray{Waters{T}}, rdx::Int, jdx::Int) where T
    cwat = crks[jdx]

    # x[old] is infeasible
    if !cwat.v
        cwat.f = fnew
        cwat.v = true
        cwat.c = 0.0

        @inbounds @simd for i in eachindex(xnew)
            cwat.x[i] = xnew[i]
        end

    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    elseif fnew ≤ rivs[rdx].f
        cwat.f = fnew

        @inbounds @simd for i in eachindex(xnew)
            cwat.x[i] = xnew[i]
        end

        swap!(wats, rdx, length(rivs) + jdx)

    # x[old], x[new] are feasible
    elseif fnew ≤ cwat.f
        cwat.f = fnew

        @inbounds @simd for i in eachindex(xnew)
            cwat.x[i] = xnew[i]
        end
    end
end

# Matchup for a feasible x[new] trial in "rivers", @code_warntype ✓
function _match!(xnew::Vector{T}, fnew::T, rivs::SubArray{Waters{T}}, rdx::Int) where T
    rwat = rivs[rdx]

    # x[old] is infeasible
    if !rwat.v
        rwat.f = fnew
        rwat.v = true
        rwat.c = 0.0

        @inbounds @simd for i in eachindex(xnew)
            rwat.x[i] = xnew[i]
        end

    # x[old], x[new] are feasible, x[new] is better than/equals to x[best] (greedy strategy)
    elseif fnew ≤ rivs[1].f
        rwat.f = fnew

        @inbounds @simd for i in eachindex(xnew)
            rwat.x[i] = xnew[i]
        end

        swap!(rivs, 1, rdx)

    # x[old], x[new] are feasible
    elseif fnew ≤ rwat.f
        rwat.f = fnew

        @inbounds @simd for i in eachindex(xnew)
            rwat.x[i] = xnew[i]
        end
    end
end

# Matchup for an infeasible x[new] trial, here "fnew == violation", @code_warntype ✓
function _match!(xnew::Vector{T}, violation::T, wat::Waters{T}) where T
    # x[old], x[new] are infeasible, compare violation
    # There is no `else` condition, if x[old] is feasible, then a matchup is unnecessary.
    if !wat.v && violation ≤ wat.c
        wat.c = violation

        @inbounds @simd for i in eachindex(xnew)
            wat.x[i] = xnew[i]
        end
    end
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Minimization/Optimization Operations of WCSCA
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#

function minimize!(fn!::F, lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {F, T, ND}
    NP   = 35 * ND
    NR   = ND + 1
    imax = 210 * ND

    minimize!(fn!, lb, ub, NP, NR, imax, 1e-7)
end

function minimize!(fn!::F, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U, NR::U, imax::U, dmax::T) where {F, T, U, ND}
    NC = NP - NR; i = 0

    wats = _waters(lb, ub, NP)   # allocations: 1 + 2 * NP
    rivs = _rivers(wats, NR)     # allocations: 1
    crks = _creeks(wats, NR, NP) # allocations: 1

    cons = Constraints(lb, ub)   # allocations: 1

    buff = Vector{T}(undef, ND)  # allocations: 1
    fork = Vector{U}(undef, NR)  # allocations: 1

    _init!(fn!, cons, wats)      # allocations: 0
    _sort!(wats)                 # allocations: 0

    sea = wats[1]

    while i < imax
        i += 1
        ss = 2.0 - _sigmoid(T(i), 0.5 * imax, 0.618, 20.0 / imax)

        _group!(fork, wats, NR, NC)

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Flowing creeks/rivers to rivers/sea
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        rdx = 1; fdx = fork[rdx]

        # flowing creeks to rivers
        @inbounds for idx in eachindex(crks)
            _trial!(buff, rivs[rdx].x, crks[idx].x, ss)
            _match!(buff, wats, rivs, crks, rdx, idx, fn!, cons)

            fdx -= 1

            if iszero(fdx)
                rdx += 1; fdx = fork[rdx]
            end
        end

        # flowing rivers to sea == rivs[1]
        @inbounds for rdx in 2:NR
            _trial!(buff, rivs[1].x, rivs[rdx].x, ss)
            _match!(buff, rivs, rdx, fn!, cons)
        end

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Raining/Evaporation process
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        @inbounds for idx in 1:fork[1]
            if_rain = _rain!(buff, sea.x, crks[idx].x, dmax)

            if if_rain
                _match!(buff, wats, rivs, crks, 1, idx, fn!, cons)
            end
        end

        @inbounds for rdx in 2:NR
            if_rain = _rain!(buff, sea.x, rivs[rdx].x, lb, ub, dmax)

            if if_rain
                _match!(buff, rivs, rdx, fn!, cons)
            end
        end

        #=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
         Renew process: update the function-value of infeasible candidates
        - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
        fmax = -Inf

        @inbounds for wat in wats
            if wat.v
                fmax = max(fmax, wat.f)
            end
        end

        @inbounds @simd for wat in wats
            if !wat.v
                wat.f = wat.c + fmax
            end
        end

        _sort!(wats); dmax -= dmax / imax
    end

    return sea
end

function minimize!(logs::WCSCALog, fn!::F, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::U, NR::U, imax::U, dmax::T) where {F, T, U, ND}
    NC = NP - NR; i = 0

    wats = _waters(lb, ub, NP)
    rivs = _rivers(wats, NR)
    crks = _creeks(wats, NR, NP)

    cons = Constraints(lb, ub)

    buff = Vector{T}(undef, ND)
    fork = Vector{U}(undef, NR)

    _init!(fn!, cons, wats)
    _sort!(wats)

    sea = wats[1]

    while i < imax
        i += 1
        ss = 2.0 - _sigmoid(T(i), 0.5 * imax, 0.618, 20.0 / imax)

        _group!(fork, wats, NR, NC)

        rdx = 1; fdx = fork[rdx]

        # flowing creeks to rivers
        @inbounds for idx in eachindex(crks)
            _trial!(buff, rivs[rdx].x, crks[idx].x, ss)
            _match!(buff, wats, rivs, crks, rdx, idx, fn!, cons)

            fdx -= 1

            if iszero(fdx)
                rdx += 1; fdx = fork[rdx]
            end
        end

        # flowing rivers to sea == rivs[1]
        @inbounds for rdx in 2:NR
            _trial!(buff, rivs[1].x, rivs[rdx].x, ss)
            _match!(buff, rivs, rdx, fn!, cons)
        end

        @inbounds for idx in 1:fork[1]
            if_rain = _rain!(buff, sea.x, crks[idx].x, dmax)

            if if_rain
                _match!(buff, wats, rivs, crks, 1, idx, fn!, cons)
            end
        end

        @inbounds for rdx in 2:NR
            if_rain = _rain!(buff, sea.x, rivs[rdx].x, lb, ub, dmax)

            if if_rain
                _match!(buff, rivs, rdx, fn!, cons)
            end
        end

        fmax = -Inf

        @inbounds for wat in wats
            if wat.v
                fmax = max(fmax, wat.f)
            end
        end

        @inbounds @simd for wat in wats
            if !wat.v
                wat.f = wat.c + fmax
            end
        end

        _sort!(wats); dmax -= dmax / imax

        # Update process: update the evolution history of WCSCA algorithm
        _update!(logs, sea, fork, i)
    end
end
