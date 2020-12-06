module VCModels

using NLopt
using LinearAlgebra
using Base: Ryu
using Distributions
using StatsBase
using StatsModels
using DataFrames
using FiniteDiff

import NLopt: Opt

export
    # Structs/constructors 
    VCData,
    VCModel,
    # Computations
    setθ!,
    updateΛ!,
    fixef,
    objective,
    fit,
    fit!,
    hessian!,
    # Utilities
    ranef,
    vcov,
    vcovvc,
    vcovvctr,
    stderror,
    stderrorvc

include("optsummary.jl")
include("vcmodel.jl")

end # module