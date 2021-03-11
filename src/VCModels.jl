module VCModels

using NLopt
using LinearAlgebra
using Base: Ryu
using Distributions
using StatsBase
using StatsModels
using DataFrames: DataFrame
using FiniteDiff

import StatsBase: fit, fit! # Can now call fit, fit! directly
#import NLopt: Opt

export
    # Structs/constructors 
    VCData,
    VCModel,
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
    hessian!,
    fisherinfo!,
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

include("vcmodel.jl")
include("optimization.jl")
end # module