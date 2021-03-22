module VCModels

using NLopt
using LinearAlgebra
using Base: Ryu
using Distributions
using StatsBase
using StatsModels
using DataFrames: DataFrame
using FiniteDiff
using LoopVectorization

import StatsBase: fit, fit! # Can now call fit, fit! directly
#import NLopt: Opt # overwrites

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
    gradient,
    gradient!,
    hessian,
    hessian!,
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
    isnested

# Order matters!
include("optimization.jl")
include("vcmodel.jl")
include("algorithms.jl")

end # module