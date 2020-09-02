export benchmark, rosen!, ackley!, rastrigin!, gen_fitdata!

function benchmark(fn!::F, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::Int=25, NR::Int=2, dmax::T=1e-7) where {F, T, ND}
    NP = NP ≥ 25 * ND ? NP : 25 * ND
    NR = NR > ND ? NR : NR = ND + 1

    imax = 150 * ND
    logs = WCSCALog(ND, NR, imax, Val(T)) # allocations: 8

    minimize!(logs, fn!, lb, ub, NP, NR, imax, 1e-7)

    return logs
end

"""
[Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)
"""
rosen!(x₁::T, x₂::T) where T = 100.0 * (x₂ - x₁^2.0)^2.0 + (1.0 - x₁)^2.0

function rosen!(x::Vector{T}) where T
    ret = 0.0

    @inbounds for i in 1:length(x) - 1
        ret += rosen!(x[i], x[i+1])
    end

    return ret
end

"""
[Ackley function](https://en.wikipedia.org/wiki/Ackley_function)
"""
function ackley!(x::Vector{T}) where T
    arg1 = 0.0
    arg2 = 0.0
    dims = length(x)

    @inbounds for i in eachindex(x)
        arg1 += x[i]^2.0
        arg2 += cospi(2.0 * x[i])
    end

    arg1 = -0.2 * sqrt(arg1 / dims)
    arg2 =  arg2 / dims

    return -20.0 * exp(arg1) - exp(arg2) + ℯ + 20.0
end

"""
[Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function)
"""
rastrigin!(x::T) where T = x^2.0 - 10.0 * cospi(2.0 * x) + 10.0

function rastrigin!(x::Vector{T}) where T
    ret = 0.0

    @inbounds for i in eachindex(x)
        ret += rastrigin!(x[i])
    end

    return ret
end

"""
Generating Fitting Data with White Noise
"""
gen_fitdata!(f::F, x::Vector{T}, p::Vector{T}) where {F, T} = gen_fitdata!(f, x, p, 0.1)

function gen_fitdata!(f::F, x::Vector{T}, p::Vector{T}, noise::T) where {F, T}
    y = Vector{T}(undef, length(x))

    gen_fitdata!(f, x, y, p, noise)

    return y
end

function gen_fitdata!(f::F, x::Vector{T}, y::Vector{T}, p::Vector{T}, noise::T) where {F, T}
    @inbounds @simd for i in eachindex(y)
        y[i] = f(x[i], p) * (1.0 + noise * rand(T))
    end
end
