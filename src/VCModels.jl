module VCModels

using NLopt
using LinearAlgebra
using Base: Ryu
using Distributions
using StatsBase
using StatsModels
using DataFrames
using FiniteDiff

#import NLopt: Opt

export
    # Structs/constructors 
    VCData,
    VCData2,
    VCModel,
    # Computations
    setθ!,
    updateΛ!,
    updateμ!,
    fixef,
    fixef!,
    objective,
    fit,
    fit!,
    hessian!,
    fisherinfo!,
    # Utilities
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
    isnested,
    f

#include("optsummary.jl")
include("vcmodel.jl")

end # module