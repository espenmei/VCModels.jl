# VCModels.jl

**VCModels.jl** is a Julia package for fitting variance component models that are structured according to known, dense, relationship matrices.
The intended use case is the analysis of genetic and environmental components of variance for quantitative traits, when genetic relationship matrices are computed from genome-wide SNP data.

## Models
**VCModels.jl** can fit models of the form
$$\boldsymbol{y} \sim N(\boldsymbol{X} \boldsymbol{\beta}, \boldsymbol{V}).$$

In this model $\boldsymbol{\beta}$ is the fixed effects of covariates $\boldsymbol{X}$ and $\boldsymbol{V}$ is the covariance matrix of the conditional responses. $\boldsymbol{V}$ can be modelled with the following structure

$$\boldsymbol{V} = \sum_{i=1}^q \delta_i \boldsymbol{R}_i$$

where $\delta_i$ are variance component parameters and $\boldsymbol{R}_i$ are symmetric matrices that are provided by the user.

The variance component parameters can be defined as functions of the vector of parameters $\boldsymbol{\theta}$ that are optimized. This may for instance be needed if $\delta_i$ is a covariance parameter that is bounded by the values of other parameters. This was the motivation for creating this package.

## Installation
The package can be installed from github with
``` julia
(@v1.6) pkg> add https://github.com/espenmei/VCModels.jl
```

## Example 1
Here is an example with simulated data showing how the package may be used to fit two variance components and two covariates for the mean.
```julia
using VCModels, LinearAlgebra, DataFrames, StatsModels

n, m = 1000, 2000
v = randn(n, m)
R1 = v * v' / m
R2 = Diagonal(ones(n))
V = 2R1 + 4R2
y = cholesky(V).L * randn(n)
dat = DataFrame(y = y, x = rand(n), z = rand(n))
r = [R1, R2]
m = fit(VCModel, @formula(y ~ 1 + x + z), dat, r)
```
This should give an output reasonably similar to this
```julia
logLik     -2 logLik  AIC        AICc       BIC
-2314.5590 4629.1180  4639.1180  4639.1784  4663.6568  

 Variance component parameters:
Comp.   Est.    Std. Error
θ₁      2.0943  -
θ₂      4.0674  -

 Fixed-effects parameters:
────────────────────────────────────────────────────
                  Coef.  Std. Error      z  Pr(>|z|)
────────────────────────────────────────────────────
(Intercept)  -0.179071     0.203213  -0.88    0.3782
x            -0.0029327    0.26284   -0.01    0.9911
z             0.435445     0.274538   1.59    0.1127
────────────────────────────────────────────────────
```
## Example 2
