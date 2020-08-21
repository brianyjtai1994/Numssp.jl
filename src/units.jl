# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# Unit Conversion
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
export @unit_convert

macro unit_convert(src, units::Expr)
    if units.head ≠ :call || units.args[1] ≠ :(=>)
        error("Invalid syntax of units converion.")
    end

    expr = Expr(:call)
    args = Vector{Any}(undef, 2)

    args[1] = Symbol(units.args[2], "_to_", units.args[3])
    args[2] = esc(src)

    expr.args = args

    return expr
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# About Energy
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
thz_to_nj(f::T)  where T<:Real = 6.626070150e-13 * f
nj_to_thz(e::T)  where T<:Real = e / 6.626070150e-13

thz_to_meV(f::T) where T<:Real = 4.135667696 * f
meV_to_thz(e::T) where T<:Real = e / 4.135667696

nm_to_nj(λ::T)   where T<:Real = 1.98644586e-7 / λ
nj_to_nm(e::T)   where T<:Real = 1.98644586e-7 / e

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

nm_to_eV(λ::T) where T<:Real = 1239.84198 / λ

function nm_to_eV(λ::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(λ)
        λ[i] = nm_to_eV(λ[i])
    end
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

eV_to_nm(e::T) where T<:Real = 1239.84198 / e

function eV_to_nm(e::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(e)
        e[i] = eV_to_nm(e[i])
    end
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# About Wavelength
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
cm⁻¹_to_nm(λ::T) where T<:Real = 10000000.0 / λ

function cm⁻¹_to_nm(λ::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(λ)
        λ[i] = cm⁻¹_to_nm(λ[i])
    end
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

nm_to_cm⁻¹(λ::T) where T<:Real = 10000000.0 / λ

function nm_to_cm⁻¹(λ::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(λ)
        λ[i] = nm_to_cm⁻¹(λ[i])
    end
end

# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
# About Speed of Light
# = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = = =
ps_to_μm(t::T) where T<:Real = round(149.896225 * t)

function ps_to_μm(t::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(t)
        t[i] = ps_to_μm(t[i])
    end
end

# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -

μm_to_ps(d::T) where T<:Real = d / 149.896225

function μm_to_ps(d::VT) where {T, VT<:AbstractArray{T}}
    @inbounds @simd for i in eachindex(d)
        d[i] = μm_to_ps(d[i])
    end
end
