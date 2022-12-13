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

## Usage

## Example