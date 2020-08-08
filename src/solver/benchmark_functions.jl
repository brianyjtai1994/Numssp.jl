function rosen!     end
function ackley!    end
function rastrigin! end

"""
[Rosenbrock function](https://en.wikipedia.org/wiki/Rosenbrock_function)
"""
rosen!(x₁::T, x₂::T) where T = 100.0 * (x₂ - x₁^2.0)^2.0 + (1.0 - x₁)^2.0

function rosen!(x::VT) where {T, VT<:AbstractVector{T}}
    ret = 0.0

    @inbounds for i in 1:length(x) - 1
        ret += rosen!(x[i], x[i+1])
    end

    return ret
end

function rosen!(xm::MT, jdx::U) where {T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in 1:size(xm, 1) - 1
        ret += rosen!(xm[i,jdx], xm[i+1,jdx])
    end

    return ret
end

function rosen!(des::Vector{T}, xm::MT) where {T, MT<:AbstractMatrix{T}}
    fill!(des, 0.0)

    @inbounds for j in axes(xm, 2), i in 1:size(xm, 1)-1
        des[j] += rosen!(xm[i,j], xm[i+1,j])
    end
end

"""
[Ackley function](https://en.wikipedia.org/wiki/Ackley_function)
"""
function ackley!(x::VT) where {T, VT<:AbstractVector{T}}
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

function ackley!(xm::MT, jdx::U) where {T, U, MT<:AbstractMatrix{T}}
    arg1 = 0.0
    arg2 = 0.0
    dims = size(xm, 1)

    @inbounds for i in axes(xm, 1)
        arg1 += xm[i,jdx]^2.0
        arg2 += cospi(2.0 * xm[i,jdx])
    end

    arg1 = -0.2 * sqrt(arg1 / dims)
    arg2 =  arg2 / dims

    return -20.0 * exp(arg1) - exp(arg2) + ℯ + 20.0
end

function ackley!(des::Vector{T}, xm::Matrix{T}) where T
    fill!(des, 0.0)
    dims = size(xm, 1)

    @inbounds for j in axes(xm, 2)
        arg1 = 0.0
        arg2 = 0.0
        for i in axes(xm, 1)
            arg1 += xm[i,j]^2.0
            arg2 += cospi(2.0 * xm[i,j])
        end

        arg1 = -0.2 * sqrt(arg1 / dims)
        arg2 =  arg2 / dims

        des[j] = -20.0 * exp(arg1) - exp(arg2) + ℯ + 20.0
    end
end

"""
[Rastrigin function](https://en.wikipedia.org/wiki/Rastrigin_function)
"""
rastrigin!(x::T) where T = x^2.0 - 10.0 * cospi(2.0 * x) + 10.0

function rastrigin!(xv::VT) where {T, VT<:AbstractVector{T}}
    ret = 0.0

    @inbounds for i in eachindex(xv)
        ret += rastrigin!(xv[i])
    end

    return ret
end

function rastrigin!(xm::MT, jdx::U) where {T, U, MT<:AbstractMatrix{T}}
    ret = 0.0

    @inbounds for i in axes(xm, 1)
        ret += rastrigin!(xm[i,jdx])
    end

    return ret
end

function rastrigin!(des::Vector{T}, xm::Matrix{T}) where T
    fill!(des, 0.0)

    @inbounds for j in axes(xm, 2), i in axes(xm, 1)
        des[j] += rastrigin!(xm[i,j])
    end
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
