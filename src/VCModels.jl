module VCModels

using Base: Ryu
using DataFrames: DataFrame
using Distributions
using FiniteDiff
using JLD
using LinearAlgebra
using NLopt
using StatsBase
using StatsModels

# Things to override
import StatsBase: fit, fit! # Can now call fit, fit! directly
import LinearAlgebra: dot, logabsdet

export
    # Structs/constructors 
    VCData,
    VCModel,
    VCOpt,
    # Computations
    update!,
    setθ!,
    updateΛ!,
    updateμ!,
    objective,
    fit,
    fit!,
    #gradient,
    #gradient!,
    hessian,
    hessian!,
    jacobian,
    jacobian!,
   # fisherinfo!,
    transform,
    # Utilities
    fixef,
    fixef!,
    ranef,
    ranef!,
    vcov,
    vcovvc,
    vcovvctr,
    stderror,
    stderrorvc,
    # Implements
    coef,
    deviance,
    dof,
    dof_residual,
    loglikelihood,
    modelmatrix,
    nobs,
    response,
    #isnested

# Order matters!
include("optimization.jl")
include("vcmodel.jl")
include("algorithms.jl")
include("linalg.jl")
include("fileIO.jl")

end # module