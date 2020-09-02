#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Linear Constraint Type Design
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
abstract type AbstractConstraint{T} <: AbstractWCSCA{T} end

struct NoConstraint{T}    <: AbstractConstraint{T} end
struct LiConstraint{T, U} <: AbstractConstraint{T} a::T; b::T; i::U end # a*x[i] + b

function Base.show(io::IO, c::LiConstraint{T}) where T
    print(io, "Linear Constraint {", T, "}:", (@sprintf "%7.2f" c.a),
          " * x[", (@sprintf "%2d" c.i), "] + (", (@sprintf "%4.1f" c.b), ")")
end

NoConstraint() = NoConstraint{Float64}()

# Constructors
resolve_lb(lb::T) where T = lb ≠ 0 ? (-abs(1.0 / lb),  1.0 * sign(lb)) : (-1.0, 0.0)
resolve_ub(ub::T) where T = ub ≠ 0 ? ( abs(1.0 / ub), -1.0 * sign(ub)) : ( 1.0, 0.0)

@generated function LiConstraint(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {ND, T}
    ex   = Expr(:tuple)
    args = Vector{Any}(undef, 2 * ND)

    @inbounds for i in 1:ND
        args[i]      = :(LiConstraint(resolve_lb(lb[$i])..., $i))
        args[i + ND] = :(LiConstraint(resolve_ub(ub[$i])..., $i))
    end

    ex.args = args

    return quote
        $(Expr(:meta, :inline))
        @inbounds return $ex
    end
end

LiConstraint(lb::Vector{T}, ub::Vector{T}) where T =
    LiConstraint(ntuple(x -> lb[x], length(lb)), ntuple(x -> ub[x], length(ub)))

# Each individual LiConstraint instance is callable, @code_warntype ✓
_constraint(c::NoConstraint{T}, v::Vector{T}) where T = 0.0
_constraint(c::LiConstraint{T}, x::T)         where T = max(0.0, c.a * x + c.b)

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Constraints Container Types Design
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
struct Constraints{NB, NL, C, T, L<:LiConstraint{T}} <: AbstractConstraint{L}
    box::NTuple{NB, L} # Linear Constraint Tuple (including lower/upper bounds)
    usr::NTuple{NL, C} # Customized Linear/Nonlinear Constraint Functions
end

Constraints(lb::NTuple{ND,T}, ub::NTuple{ND,T}) where {C, ND, T} =
    Constraints(LiConstraint(lb, ub), (NoConstraint(),))
Constraints(lb::NTuple{ND,T}, ub::NTuple{ND,T}, usr::NTuple{NL,C}) where {C, ND, NL, T} =
    Constraints(LiConstraint(lb, ub), usr)

_constraint(c::Constraints, xnew::Vector{T}) where T = _constraint(c.box, c.usr, xnew)

function _constraint(box::NTuple{NB,L}, usr::NTuple{NL,C}, xnew::Vector{T}) where {NB, NL, L, C, T}
    violation = 0.0

    @inbounds for i in eachindex(box)
        constraint = box[i]
        violation += _constraint(constraint, xnew[constraint.i])
    end

    @inbounds for i in eachindex(usr)
        constraint = usr[i]
        violation += _constraint(constraint, xnew)
    end

    return violation
end
