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
Here is an example with simulated data showing how the package may be used to fit two variance components and two covariates for the means. The model for the means can be defined using the `@formula` macro together with a `DataFrame` storing the covariates and the response variable. The model for the covariance can be defined by providing a vector with relationship matrices.
```julia
using VCModels, LinearAlgebra, DataFrames, StatsModels

n, m = 1000, 2000
v = randn(n, m)
R1 = Symmetric(v * v' / m)
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

Not all arguments are available when fitting a model this way. Sometimes it may be useful to first define a model with the relevant arguments and in a next step to fit that model.

```julia
y = dat[!,:y]
X = Matrix(hcat(ones(n), dat[!,2:3]))
dat = VCData(y, X, r)
m2 = VCModel(dat, [1.0, 1.0], [0.0, 0.0])
fit!(m2)
```
This first creates an object of type `VCData` that is used to hold the data input to a `VCModel`. Then a model, represented by an object of type `VCModel` is created from a constructor that accepts a `VCData`, a vector of initial values for the variance component parameters `[1.0, 1.0]` and a vector of lower bounds for those parameters `[0.0, 0.0]`. And in the last line that model is fitted `fit!(m2)`. The `!` signals that the function modifies the model, which is a convention in julia.

## Example 2
Consider the model for responses obtained from offspring
$$\boldsymbol{y} = \boldsymbol{X} \boldsymbol{\beta} + \boldsymbol{o} + \boldsymbol{m} + \boldsymbol{p} + \boldsymbol{\epsilon},$$
where $\boldsymbol{o}$ are direct additive effects of offspring own genes, $\boldsymbol{m}$ and $\boldsymbol{p}$ are indirect effects of maternal and paternal genes and $\boldsymbol{\epsilon}$ errors. The direct and indirect genetic effects may be correlated among each other, and they are correlated among individuals according to their genetic relatedness. The covariance matrix of the responses can be described as
$$\boldsymbol{V}=\sigma^2_{o}\boldsymbol{A}_{oo}+\sigma^2_{m}\boldsymbol{A}_{mm}+\sigma^2_{p}\boldsymbol{A}_{pp}+\sigma_{om}(\boldsymbol{A}_{om}+\boldsymbol{A}_{mo})+\sigma_{op}(\boldsymbol{A}_{op}+\boldsymbol{A}_{po})+\sigma_{mp}(\boldsymbol{A}_{mp}+\boldsymbol{A}_{pm})+\sigma^2_{\epsilon}\boldsymbol{I}.$$
The genetic relationship matrices $\boldsymbol{A}_{ij}$ will be estimated from a simulated dataset of 10000 SNPs from 6000 parent-offspring trios stored in *plink* format. The julia package `SnpArrays` is used to read snp-data and compute genetic relationships.

```julia
using VCModels, LinearAlgebra, DataFrames, StatsModels, SnpArrays

gpath = joinpath(@__DIR__, "data", "trio")
trio = SnpData(gpath)
A = 2 * grm(trio.snparray)
```
Because the snp-data was arranged according to families within roles, The genetic relationship matrix can be partitioned as
