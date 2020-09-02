# Numssp

`Numssp.jl` is a numerical package for the daily routines of my solid state physics (SSP) experiments.

## 1. Optimization and Root Finding

Based on the implementation of the [WCSCA](src/solver/README.md) solver, the following functionality is provided.

### 1.1 Function Minimization (Root Finding)

```julia
minimize!(fn!::Function, lb::NTuple{ND,T}, ub::NTuple{ND,T}[, NP::Int, NR::Int, imax::Int, dmax::T])
```
where:
- `fn!` is the objective function to be minimized.
- `lb` and `ub` are the lower/upper boundaries to define the feasible region to be minimized.
- `NP` is the popuplation size for WCSCA (*optional*).
- `NR` is the number of splitting in two subpopulations (*optional*).
- `imax` is the maximum iteration of WCSCA evolution (*optional*).
- `dmax` is the criterion distance between candidates to start a stagnation treatment (*optional*).

The objective function `fn!` should be defined as:
```julia
function fn!(params::Vector{T}) where T
    # body ...
end
```

A simple way to conduct a minimization is:
```julia
res = minimize!(fn!, lb, ub)
```
and use:
- `res.x` to access the optiomized parameters.
- `res.f` to access the corresponding function value.

### 1.2 Least-Square and χ² Curve Fitting

A χ² curve fitting is described as:

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=\chi^2&space;=&space;\sum_{i=1}^{N}&space;\left(&space;\dfrac{y_i&space;-&space;f(x_i;&space;\vec{p})}{\sigma_i}&space;\right)^2&space;&plus;&space;c" target="_blank"><img src="https://latex.codecogs.com/svg.latex?\chi^2&space;=&space;\sum_{i=1}^{N}&space;\left(&space;\dfrac{y_i&space;-&space;f(x_i;&space;\vec{p})}{\sigma_i}&space;\right)^2&space;&plus;&space;c" title="\chi^2 = \sum_{i=1}^{N} \left( \dfrac{y_i - f(x_i; \vec{p})}{\sigma_i} \right)^2 + c" /></a></div>

, when all of `σ` are `1.0` is a least-squrea curve fitting.

To conduct:
- χ² curve fitting by passing `(xdat, ydat, σdat, lb, ub)`.
- least-square curve fitting by passing `(xdat, ydat, lb, ub)`.

where
- `xdat::Vector{T}` is the x-raw data
- `ydat::Vector{T}` is the y-raw data
- `σdat::Vector{T}` is the σ-raw data

Available fitting models are described as below:

#### Multi-Exponential Decay

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=f(x)&space;=&space;A_1&space;\cdot&space;e^{-x/\tau_1}&space;&plus;&space;A_2&space;\cdot&space;e^{-x/\tau_2}&space;&plus;&space;\ldots&space;&plus;&space;c" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f(x)&space;=&space;A_1&space;\cdot&space;e^{-x/\tau_1}&space;&plus;&space;A_2&space;\cdot&space;e^{-x/\tau_2}&space;&plus;&space;\ldots&space;&plus;&space;c" title="f(x) = A_1 \cdot e^{-x/\tau_1} + A_2 \cdot e^{-x/\tau_2} + \ldots + c" /></a></div>

```julia
res = decay_fit(xdat, ydat[, σdat], lb, ub)
```

#### Gaussian Distribution

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=f_{\text{G}}(x;&space;A,&space;\mu,&space;\sigma,&space;c)&space;=&space;\dfrac{A}{\sigma&space;\sqrt{2\pi}}&space;\cdot&space;e^{-&space;\frac{1}{2}\left(\frac{x&space;-&space;\mu}{\sigma}\right)^2}&space;&plus;&space;c" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{\text{G}}(x;&space;A,&space;\mu,&space;\sigma,&space;c)&space;=&space;\dfrac{A}{\sigma&space;\sqrt{2\pi}}&space;\cdot&space;e^{-&space;\frac{1}{2}\left(\frac{x&space;-&space;\mu}{\sigma}\right)^2}&space;&plus;&space;c" title="f_{\text{G}}(x; A, \mu, \sigma, c) = \dfrac{A}{\sigma \sqrt{2\pi}} \cdot e^{- \frac{1}{2}\left(\frac{x - \mu}{\sigma}\right)^2} + c" /></a></div>

```julia
res = gauss_fit(xdat, ydat[, σdat], lb, ub)
```

#### Lorentzian Distribution

<div align=center><a href="https://www.codecogs.com/eqnedit.php?latex=f_{\text{L}}(x;&space;A,&space;\mu,&space;\Gamma,&space;c)&space;=&space;\dfrac{A}{\pi}&space;\cdot&space;\dfrac{\Gamma&space;/&space;2}{(x&space;-&space;\mu)^2&space;&plus;&space;(\Gamma&space;/&space;2)^2}&space;&plus;&space;c" target="_blank"><img src="https://latex.codecogs.com/svg.latex?f_{\text{L}}(x;&space;A,&space;\mu,&space;\Gamma,&space;c)&space;=&space;\dfrac{A}{\pi}&space;\cdot&space;\dfrac{\Gamma&space;/&space;2}{(x&space;-&space;\mu)^2&space;&plus;&space;(\Gamma&space;/&space;2)^2}&space;&plus;&space;c" title="f_{\text{L}}(x; A, \mu, \Gamma, c) = \dfrac{A}{\pi} \cdot \dfrac{\Gamma / 2}{(x - \mu)^2 + (\Gamma / 2)^2} + c" /></a></div>

```julia
res = lorentz_fit(xdat, ydat[, σdat], lb, ub)
```

## 2. Unit Conversion

A functionality of unit conversion, by using
```julia
@unit_convert x "unit_1"=>"unit_2"
```
where `x` can be any `<:Real` number or `AbstractVector{<:Real}`. Currently, the following conversion is provided:
- `"thz", "nj"`: terahertz <=> nanojoul
- `"thz", "meV"`: terahertz <=> milli-electron volt
- `"nm", "nj"`: nanometer <=> nanojoul
- `"nm", "eV"`: nanometer <=> electron volt
- `"cm⁻¹", "nm"`: reciprocal wavelength <=> nanometer
- `"ps", "μm"`: picosecond <=> micrometer

## 3. Some Customized Function for [PyPlot](https://github.com/JuliaPy/PyPlot.jl.git)

- For setting of [using LaTeX](https://matplotlib.org/tutorials/text/usetex.html?highlight=usetex) and fontsize:
    ```julia
    using PyPlot
    set_rcParams!(rc, PyPlot.PyDict(PyPlot.matplotlib."rcParams"), fontsize=12)
    ```

- For setting ticklabels of specific direction:
    ```julia
    set_ticklabels!(axis, tick_start, tick_end, tick_step, direction, pad=true)
    ``` 
    where `tick_start:tick_step:tick_end` can be both `Int` or `Float64`, `direction` should be `:x, :y, :z`.

    If you have used
    ```julia
    axis[:set_xticks](..., ...)
    axis[:set_yticks](..., ...)
    axis[:set_zticks](..., ...)
    ```
    then `pad` should be passed by `false`.

- For saving a pdf format figure:
    ```julia
    save_pdf(file_name, fig)
    ```
