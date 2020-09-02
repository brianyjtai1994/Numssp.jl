# @code_warntype ✓
function swap!(v::AbstractVector{T}, i::Int, j::Int) where T
    v[i], v[j] = v[j], v[i]

    return nothing # to avoid a tuple-allocation
end

function insertion_sort!(arr::AbstractVector{T}, ldx::Int, rdx::Int) where T
    @inbounds for idx in ldx+1:rdx
        val = arr[idx]
        jdx = idx

        while jdx > ldx && val < arr[jdx - 1]
            swap!(arr, jdx, jdx - 1); jdx -= 1
        end
    end
end

bilast_insert(arr::AbstractVector{T}, val::T) where T = bilast_insert(arr, val, 1, length(arr))

function bilast_insert(arr::AbstractVector{T}, val::T, ldx::Int, rdx::Int) where T
    if ldx ≥ rdx
        return ldx
    end

    upper_bound = rdx

    @inbounds while ldx < rdx
        mdx = (ldx + rdx) >> 1

        if val < arr[mdx]
            rdx = mdx
        else
            ldx = mdx + 1 # arr[mdx].f == val in this case
        end
    end

    if ldx == upper_bound && arr[ldx] ≤ val
        ldx += 1
    end

    return ldx
end

binary_insertsort!(arr::AbstractVector{T}) where T = binary_insertsort!(arr, 1, length(arr))

function binary_insertsort!(arr::AbstractVector{T}, ldx::Int, rdx::Int) where T
    @inbounds for idx in ldx+1:rdx
        val = arr[idx]
        loc = bilast_insert(arr, val, ldx, idx)

        jdx = idx

        while jdx > loc
            swap!(arr, jdx, jdx - 1); jdx -= 1
        end
    end
end
