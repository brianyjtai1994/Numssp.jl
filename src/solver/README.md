# Water-Cycle Sine-Cosine Algorithm (WCSCA)

Water-Cycle Sine-Cosine Algorithm (WCSCA) is a hybrid population-based algorithm based on [water-cycle algorithm](http://dx.doi.org/10.1016/j.compstruc.2012.07.010) and [sine-cosine algorithm](http://dx.doi.org/10.1016/j.knosys.2015.12.022) with constraint handling method based on [this article](https://doi.org/10.1016/S0045-7825(99)00389-8). This global optimization algorithm is used to solve curve/model fitting problems without lots of annoying initial assumptions of fitting parameters.

Generally, similar to other population-based algorithm, you only have to provide two things:
- objective function `fn!`
- fitting lower/upper boundaries `lb, ub`

## 1. Benchmark

To benchmark the capability of WCSCA global optimization, a function is already provided:
```julia
benchmark(fn!::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}, NP::Int, NR::Int, dmax::T)
```
where:
- `fn!` is the objective function to be minimized. Basically, there are 3 common benchmark functions provided by `Numssp.jl`
    - [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) as `rosen!`
    - [Ackley](https://en.wikipedia.org/wiki/Ackley_function) as `ackley!`
    - [Rastrigin](https://en.wikipedia.org/wiki/Rastrigin_function)  as `rastrigin!`
- `lb` and `ub` are the lower/upper boundaries to define the feasible region to be minimized.
- `NP` is the popuplation size for WCSCA (*optional*).
- `NR` is the number of splitting in two subpopulations (*optional*).
- `dmax` is the criterion distance between candidates to start a stagnation treatment (*optional*).

A simple way to conduct a benchmark is:
```julia
using Numssp, PyPlot

fn! = rosen!; ND = 30

lbv = -30.; ubv = 30.

lb = ntuple(x -> lbv, ND)
ub = ntuple(x -> ubv, ND)

history = benchmark(fn!, lb, ub)
```

The function `benchmark` will return a `WCSCALog` type which records the history of the WCSCA evolution. A series of plots can be made to see how the WCSCA works like below:
```julia
ND, NR, imax = history.dims

x = [1:imax;]

fg = figure(figsize=[12.8, 3.6 * ND], clear=true)

ax = fg[:add_subplot](NR, 2, 1)
ax[:plot](x, history.fsol, lw=1.5)
ax[:set_yscale]("symlog")
ax[:set_xscale]("log")
ax[:set_xlim](x[1], x[end])
ax[:set_title](string(history.fsol[end]))

for k in 1:ND
    ax = fg[:add_subplot](NR, 2, 2k+1)
    ax[:plot](x, history.xsol[k, :], lw=1.5, c=string("C", k))
    ax[:set_ylim](lbv, ubv)
    ax[:set_yscale]("symlog")
    ax[:set_xscale]("log")
    ax[:set_xlim](x[1], x[end])
    ax[:set_title](string(history.xsol[k, end]))
end

for k in 1:NR
    ax = fg[:add_subplot](NR, 2, 2k)
    ax[:plot](x, history.fork[k, :], lw=1.5, c=string("C", k-1))
    ax[:set_xscale]("log")
    ax[:set_xlim](x[1], x[end])
end

fg[:set_tight_layout](true)
```
## 2. Sorting Algorithm

A binary insertion sorting algorithm, as a variant of common [insertion sort](https://en.wikipedia.org/wiki/Insertion_sort), is a *stable* sorting used in WCSCA to sort the whole popuplation. To maintain the stability of sorting, a [rightmost binary search](https://en.wikipedia.org/wiki/Binary_search_algorithm) is used:
```julia
function bilast(arr::AbstractVector{T}, val::T, ldx::Int, rdx::Int) where T
    if ldx ≥ rdx
        return ldx
    end

    upper_bound = rdx

    @inbounds while ldx < rdx
        mdx = (ldx + rdx) >> 1

        if val < arr[mdx]
            rdx = mdx
        else
            ldx = mdx + 1
        end
    end

    if ldx == upper_bound && arr[ldx] ≤ val
        ldx += 1
    end

    return ldx
end
```
