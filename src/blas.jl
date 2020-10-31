#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Given a matrix `A`, this routine 1st performs a LUP decomposition and store in a buffer `LU`.

 Partial pivoting is applied so after the 1st step, the transformation is
     A⋅x = b -> LU⋅x = P⋅b, where P⋅A -> LU
 Then, the 2nd step is forward substitution for solving
     L⋅y = P⋅b
 Finally, the 3rd step is back substitution for solving
     U⋅x = y

 Test:
     A = [0. 14.  9.  3.  5.;
          1.  0. -5.  2. -3.;
          9. 15.  0.  5. 16.;
          3.  2.  5.  0. 49.;
          5. 32. 16. 49.  0.]

     b = [61., -17., 85., 205., 206.]

 Ans.:
     x = [-1., 1., 2., 3., 4.]

 Reference:
     T. H. Cormen et al., Introduction to Algorithms 3rd edition, p824
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
# @code_warntype ✓
function lup!(lu::Matrix{T}, p::Vector{Int}, A::AbstractMatrix{T}) where T
    @inbounds @simd for idx in eachindex(lu)
        lu[idx] = A[idx]
    end

    return lup!(lu, p)
end

# @code_warntype ✓
function lup!(lu::Matrix{T}, p::Vector{Int}) where T
    nrow, ncol = size(lu)

    @inbounds begin
        p[end] = ncol

        for kdx in 1:ncol # column-wise
            kpiv = kdx # kth column pivot

            # find pivoting index
            amax = 0.0 # max. abs. of kth column
            for idx in kdx:nrow
                temp = abs(lu[idx, kdx])
                if temp > amax
                    kpiv = idx
                    amax = temp
                end
            end

            p[kdx] = kpiv

            # check to do row-swap or not
            if !iszero(amax) # A[pivot] is not zero
                if kdx ≠ kpiv
                    # interchange
                    @simd for jdx in 1:ncol
                        lu[kdx, jdx], lu[kpiv, jdx] = lu[kpiv, jdx], lu[kdx, jdx]
                    end

                    p[end] += 1
                end

                # scale first column
                lukkinv = inv(lu[kdx, kdx])

                if lukkinv == zero(T)
                    lukkinv = 1e-40
                end

                @simd for idx in kdx+1:nrow
                    lu[idx, kdx] *= lukkinv
                end

            # matrix is sigular if pivot element is zero
            else
                return false # Error: "Sigular matrix"
            end

            # update the rest
            for jdx in kdx+1:ncol, idx in kdx+1:nrow
                lu[idx, jdx] -= lu[idx, kdx] * lu[kdx, jdx]
            end
        end
    end

    return true
end

# LUP solving linear equations and store results in `x`, @code_warntype ✓
function la_solve!(x::AbstractVector{T}, A::AbstractMatrix{T}, lu::Matrix{T}, p::Vector{Int}) where T
    return lup!(lu, p, A) ? la_solve!(x, lu, p) : false
end

function la_solve!(x::AbstractVector{T}, lu::Matrix{T}, p::Vector{Int}) where T
    nrow, ncol = size(lu)

    @inbounds begin
        for idx in eachindex(x)
            pidx = p[idx]
            x[idx], x[pidx] = x[pidx], x[idx]
        end

        # forward substitution
        for idx in 2:length(x), jdx in 1:idx-1
            x[idx] -= lu[idx, jdx] * x[jdx]
        end

        # back substitution
        x[end] /= lu[nrow, ncol]

        for idx in nrow-1:-1:1
            for jdx in idx+1:ncol
                x[idx] -= lu[idx, jdx] * x[jdx]
            end

            x[idx] /= lu[idx, idx]
        end
    end

    return true
end

function det(lu::Matrix{T}, p::Vector{Int}) where T
    ret = one(T); N = length(p) - 1

    @inbounds for i in 1:N
        ret *= lu[i, i]
    end

    return Bool((p[end] - N) & 1) ? -ret : ret
end

#=
 Do matrix inversion of `src` and stored in `des`.
 `lu`, `p` are provide buffers for LUP and pivoting
 @code_warntype ✓
=#
function inv!(des::AbstractMatrix{T}, src::AbstractMatrix{T}, lu::Matrix{T}, p::Vector{Int}) where T
    nrow, ncol = size(src)

    if nrow ≠ ncol
        error("`src` SHOULD BE a sqaure matrix!")
    end

    if size(des) ≠ size(src)
        error("Sizes of `des` and `src` SHOULD BE MATCHED!")
    end

    lup!(lu, p, src)

    @inbounds for jdx in 1:ncol
        @simd for idx in 1:nrow
            des[idx, jdx] = ifelse(idx == jdx, one(T), zero(T))
        end

        la_solve!(view(des, :, jdx), lu, p)
    end
end

#=- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
 Matrix/Matrix & Matrix/Vector Multiplication
 - nAXnB : A ⋅B
 - tAXnB : A'⋅B
 - nAXnV : A ⋅V
 - tAXnV : A'⋅V
 - tVXnV : V'⋅V
 - nVXtV : V ⋅V'

 - tVXnAXnV : V'⋅A⋅V
- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -=#
function nAXnB!(matC::AbstractMatrix{T}, matA::AbstractMatrix{T}, matB::AbstractMatrix{T}) where T
    # matA ∈ m*n, matB ∈ n*p, matC ∈ m*p
    m, p = size(matC); n = size(matA, 2)

    @inbounds for jdx in 1:p, idx in 1:m
        temp = zero(T)

        for kdx in 1:n
            temp += matA[idx, kdx] * matB[kdx, jdx]
        end

        matC[idx, jdx] = temp
    end
end

function tAXnB!(matC::AbstractMatrix{T}, matA::AbstractMatrix{T}, matB::AbstractMatrix{T}) where T
    # matA ∈ m*n, matB ∈ n*p, matC ∈ m*p
    m, p = size(matC); n = size(matA, 1)

    @inbounds for jdx in 1:p, idx in 1:m
        temp = zero(T)

        for kdx in 1:n
            temp += matA[kdx, idx] * matB[kdx, jdx]
        end

        matC[idx, jdx] = temp
    end
end

function nAXnV!(vecC::AbstractVector{T}, matA::AbstractMatrix{T}, vecB::AbstractVector{T}) where T
    nrow, ncol = size(matA)

    @inbounds for idx in 1:nrow
        temp = zero(T)

        for jdx in 1:ncol
            temp += matA[idx, jdx] * vecB[jdx]
        end

        vecC[idx] = temp
    end
end

function tAXnV!(vecC::AbstractVector{T}, matA::AbstractMatrix{T}, vecB::AbstractVector{T}) where T
    nrow, ncol = size(matA)

    @inbounds for idx in 1:ncol
        temp = zero(T)

        for jdx in 1:nrow
            temp += matA[jdx, idx] * vecB[jdx]
        end

        vecC[idx] = temp
    end
end

function tVXnV(vecA::AbstractVector{T}, vecB::AbstractVector{T}) where T
    ret = zero(T)

    @inbounds for idx in eachindex(vecA)
        ret += vecA[idx] * vecB[idx]
    end

    return ret
end

function nVXtV!(matC::AbstractMatrix{T}, vecA::AbstractVector{T}, vecB::AbstractVector{T}) where T
    nrow = length(vecA); ncol = length(vecB)

    @inbounds for jdx in 1:ncol
        @simd for idx in 1:nrow
            matC[idx, jdx] = vecA[idx] * vecB[jdx]
        end
    end
end

function tVXnAXnV(v::AbstractVector, m::AbstractMatrix{T}) where T
    n = length(v); ret = zero(T)

    @inbounds for jdx in 1:n, idx in 1:n
        ret += v[idx] * m[idx, jdx] * v[jdx]
    end

    return ret
end
