export fft, unwrap!, fftfreq

fft(x::Vector{T}) where T = fft(x, 1 << ceil(Int, log2(length(x))))

function fft(x::Vector{T}, n::U) where {T, U}
    if Bool(n & (n-1))
        error("fft size must be radix-2!")
    end

    r, a, b = difnn(x, n)

    if r
        circshift!(a, b, n >> 1); return a
    else
        circshift!(b, a, n >> 1); return b
    end
end

function twiddle(H::Int)
    ret = Vector{ComplexF64}(undef, H)

    @inbounds @simd for i in eachindex(ret)
        ret[i] = cospi((i - 1) / H) - im * sinpi((i - 1) / H)
    end

    return ret
end

function difbutfly!(w::VT, x::VT, y::VT, H::U, h::U, s::U, S::U, j::U, k::U; jw::U=1) where {T, U, VT<:Vector{Complex{T}}}
    @inbounds for _ in 1:h
        y[j]   = (x[k] + x[k+H])
        y[j+s] = (x[k] - x[k+H]) * w[jw]
        j += S; jw += s; k += s
    end
end

function difnn(x::Vector{T}, n::U) where {T, U}
    a = Vector{Complex{T}}(undef, n)
    b = Vector{Complex{T}}(undef, n)
    H = n >> 1; w = twiddle(H)
    h = n >> 1; s = 1; S = 2; r = false

    copyto!(a, x)

    while h > 0
        (x, y) = r ? (b, a) : (a, b)

        for j₀ in 1:s
            difbutfly!(w, x, y, H, h, s, S, j₀, j₀)
        end

        h >>= 1; s <<= 1; S <<= 1; r = !r
    end

    return r, a, b
end

function unwrap!(x::Vector{T}) where T
    xᵢ₋₁   = x[1]
    period = 0.0
    Δphase = 0.0

    @inbounds for i in 2:length(x)
        Δphase = x[i] - xᵢ₋₁

        if Δphase > π
            period -= 1.0

        elseif Δphase < -π
            period += 1.0
        end

        xᵢ₋₁ = x[i]
        x[i] = xᵢ₋₁ + 2.0π * period
    end

    return x
end

function fftfreq(t::Vector{T}, N::U) where {T, U}
    fq = similar(t)
    N½ = N >> 1 + 1
    Δf = 1.0 / (t[end] - t[1])

    @inbounds @simd for i in eachindex(fq)
        fq[i] = Δf * (i - N½)
    end

    return fq
end
