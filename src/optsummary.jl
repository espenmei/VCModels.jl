# this is a comment from Eirik
mutable struct OptSummary{T<:AbstractFloat}
    initial::Vector{T}
    lowerbd::Vector{T}
    finitial::T
    ftol_rel::T
    ftol_abs::T
    xtol_rel::T
    xtol_abs::T
    initial_step::Vector{T}
    maxfeval::Int
    final::Vector{T}
    fmin::T
    feval::Int
    optimizer::Symbol
    returnvalue::Symbol
end

function OptSummary(initial::Vector{T}, lowerbd::Vector{T}) where {T<:AbstractFloat}
    OptSummary(
        initial,
        lowerbd,
        T(Inf),
        T(1.0e-12),
        T(1.0e-8),
        zero(T),
        T(1.0e-10),
        zero(initial),
        -1,
        copy(initial),
        T(Inf),
        -1,
        :LN_BOBYQA,
        :FAILURE
    )
end

# Overwrite NLopt.opt
function NLopt.Opt(optsum::OptSummary)
    lb = optsum.lowerbd
    opt = NLopt.Opt(optsum.optimizer, length(lb))
    NLopt.ftol_rel!(opt, optsum.ftol_rel) # relative criterion on objective
    NLopt.ftol_abs!(opt, optsum.ftol_abs) # absolute criterion on objective
    NLopt.xtol_rel!(opt, optsum.xtol_rel) # relative criterion on parameter values
    NLopt.xtol_abs!(opt, optsum.xtol_abs) # absolute criterion on parameter values
    NLopt.lower_bounds!(opt, lb) # lower bounds
    NLopt.maxeval!(opt, optsum.maxfeval) # maximum number of function evaluations
    NLopt.initial_step(opt, optsum.initial, similar(lb))
    opt
end
