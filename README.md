# Numssp

`Numssp` is a numerical package for the daily routines of my solid state physics (SSP) experiments.

## 1. Water-Cycle Sine-Cosine Algorithm (WCSCA)

Water-Cycle Sine-Cosine Algorithm (WCSCA) is a hybrid population-based algorithm based on [water-cycle algorithm](http://dx.doi.org/10.1016/j.compstruc.2012.07.010) and [sine-cosine algorithm](http://dx.doi.org/10.1016/j.knosys.2015.12.022) with constraint handling method based on [this article](https://doi.org/10.1016/S0045-7825(99)00389-8). This algorithm is used to solve curve/model fitting problems without lots of annoying initial assumptions of fitting parameters. Generally, similar to other population-based algorithm, you only have to provide two things: objective function `fn!` and fitting lower/upper boundaries `lb, ub`.

It will be easy to start with:
```julia
using Numssp, PyPlot
```
And there are 3 pre-defined common models for benchmark: [Rosenbrock](https://en.wikipedia.org/wiki/Rosenbrock_function) as `rosen!`, [Ackley](https://en.wikipedia.org/wiki/Ackley_function) as `ackley!`, and [Rastrigin](https://en.wikipedia.org/wiki/Rastrigin_function)  as `rastrigin!`. Welcome to test if the WCSCA is able to solve a high-dimensional optimization problem like following:
```julia
fn! = rosen!; ND = 30

lbv = -30.; ubv = 30.

lb = ntuple(x -> lbv, ND)
ub = ntuple(x -> ubv, ND)

history = evolve_benchmark(fn!, lb, ub)
```
The function `evolve_benchmark` will return a `WCSCALog` type which records the history of the WCSCA evolution. You can make a series of plots to see how the WCSCA works like below:
```julia
ND, NR, it_max = history.log_dims

x = [1:it_max;]

fg = figure(figsize=[12.8, 3.6 * ND], clear=true)

ax = fg.add_subplot(NR, 2, 1)
ax.plot(x, history.log_fsol, lw=1.5)
ax.set_yscale("symlog")
ax.set_xscale("log")
ax.set_xlim(x[1], x[end])
ax.set_title(string(history.log_fsol[end]))

for k in 1:ND
    ax = fg.add_subplot(NR, 2, 2k+1)
    ax.plot(x, history.log_xsol[k, :], lw=1.5, c=string("C", k))
    ax.set_ylim(lbv, ubv)
    ax.set_yscale("symlog")
    ax.set_xscale("log")
    ax.set_xlim(x[1], x[end])
    ax.set_title(string(history.log_xsol[k, end]))
end

for k in 1:NR
    ax = fg.add_subplot(NR, 2, 2k)
    ax.plot(x, history.log_fork[k, :], lw=1.5, c=string("C", k-1))
    ax.set_xscale("log")
    ax.set_xlim(x[1], x[end])
end

fg.set_tight_layout(true)
```

In addition, based on WCSCA solver, the following functionality is provided.

### 1.1 Multi-Exponential Decay

The multi-exponential decay provided by `Numssp` is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;A_1&space;\cdot&space;e^{-x/\tau_1}&space;&plus;&space;A_2&space;\cdot&space;e^{-x/\tau_2}&space;&plus;&space;\ldots&space;&plus;&space;c" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;A_1&space;\cdot&space;e^{-x/\tau_1}&space;&plus;&space;A_2&space;\cdot&space;e^{-x/\tau_2}&space;&plus;&space;\ldots&space;&plus;&space;c" title="f(x) = A_1 \cdot e^{-x/\tau_1} + A_2 \cdot e^{-x/\tau_2} + \ldots + c" /></a>

where the parameters should be arranged in a `NTuple` form as `(A₁, τ₁, A₂, τ₂, ..., c)` and be passed to the `decay_fit` function like
```julia
params, _ = decay_fit(xdat, ydat, lb, ub)
```
where `xdat::Vector{T}` and `ydat::Vector{T}` are the raw datas according to the above formula, and the `params` is the solution.

### 1.2 Gaussian Distribution

The Gaussian distribution provided by `Numssp` is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{\text{G}}(x;&space;A,&space;\mu,&space;\sigma,&space;c)&space;=&space;\dfrac{A}{\sigma&space;\sqrt{2\pi}}&space;\cdot&space;e^{-&space;\frac{1}{2}\left(\frac{x&space;-&space;\mu}{\sigma}\right)^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{\text{G}}(x;&space;A,&space;\mu,&space;\sigma,&space;c)&space;=&space;\dfrac{A}{\sigma&space;\sqrt{2\pi}}&space;\cdot&space;e^{-&space;\frac{1}{2}\left(\frac{x&space;-&space;\mu}{\sigma}\right)^2}" title="f_{\text{G}}(x; A, \mu, \sigma, c) = \dfrac{A}{\sigma \sqrt{2\pi}} \cdot e^{- \frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2}" /></a>

and is easy to call:
```julia
params, _ = gauss_fit(xdat, ydat, lb, ub)
```

### 1.3 Lorentzian Distribution

The Lorentzian distribution provided by `Numssp` is defined as:

<a href="https://www.codecogs.com/eqnedit.php?latex=f_{\text{L}}(x;&space;A,&space;\mu,&space;\Gamma,&space;c)&space;=&space;\dfrac{A}{\pi}&space;\cdot&space;\dfrac{\Gamma&space;/&space;2}{(x&space;-&space;\mu)^2&space;&plus;&space;(\Gamma&space;/&space;2)^2}" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{\text{L}}(x;&space;A,&space;\mu,&space;\Gamma,&space;c)&space;=&space;\dfrac{A}{\pi}&space;\cdot&space;\dfrac{\Gamma&space;/&space;2}{(x&space;-&space;\mu)^2&space;&plus;&space;(\Gamma&space;/&space;2)^2}" title="f_{\text{L}}(x; A, \mu, \Gamma, c) = \dfrac{A}{\pi} \cdot \dfrac{\Gamma / 2}{(x - \mu)^2 + (\Gamma / 2)^2}" /></a>

and is easy to call:
```julia
params, _ = lorentz_fit(xdat, ydat, lb, ub)
```

### 1.4 χ² Curve Fitting (Experimental)

It is also possible to have a χ² curve fitting or weighted least square fitting according to:

<a href="https://www.codecogs.com/eqnedit.php?latex=\chi^2&space;=&space;\sum_{i=1}^{N}&space;\left(&space;\dfrac{y_i&space;-&space;f(x_i;&space;\vec{p})}{\sigma_i}&space;\right)^2" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\chi^2&space;=&space;\sum_{i=1}^{N}&space;\left(&space;\dfrac{y_i&space;-&space;f(x_i;&space;\vec{p})}{\sigma_i}&space;\right)^2" title="\chi^2 = \sum_{i=1}^{N} \left( \dfrac{y_i - f(x_i; \vec{p})}{\sigma_i} \right)^2" /></a>

and is easy to call:
```julia
params, _ = xxx_fit(xdat, ydat, σdat, lb, ub)
```
, where `xxx` stands for the name of distribution in `decay, gauss, lorentz`, and the elements in `σdat::Vector{T}` should be `σ⁻²`.
