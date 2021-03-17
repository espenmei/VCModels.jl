
#- `reml`: boolean indicator for reml
mutable struct VCOpt{T<:AbstractFloat}
    optimizer::Symbol
    xlb::Vector{T}
    xtol_abs::T
    xtol_rel::T
    ftol_abs::T
    ftol_rel::T
    feval::Int
    maxfeval::Int
    xinitial::Vector{T}
    xfinal::Vector{T}
    finitial::T
    ffinal::T
    reml::Bool
    H::Matrix{Union{Missing, T}}
    ∇::Vector{Union{Missing, T}}
end

function VCOpt(optimizer::Symbol, x::Vector{T}, xlb::Vector{T}, reml::Bool = false) where {T<:AbstractFloat}
    q = length(xlb)
    VCOpt(
        optimizer,
        xlb,
        T(10^-10),
        zero(T),
        T(10^-8),
        T(10^-12),
        -1,
        -1,
        x,
        zeros(T, q),
        T(0),
        T(0),
        reml,
        Matrix{Union{Missing, T}}(missing, q, q),
        Vector{Union{Missing, T}}(missing, q)
    )
end

function update!(o::VCOpt, xinitial::Vector, finitial::T) where T<:AbstractFloat
    o.xinitial = x.initial
    o.finitial = finitial
    o
end

function NLopt.Opt(o::VCOpt)
    opt = NLopt.Opt(o.optimizer, length(o.xlb))
    NLopt.lower_bounds!(opt, o.xlb)
    NLopt.xtol_abs!(opt, o.xtol_abs)
    NLopt.xtol_rel!(opt, o.xtol_rel)
    NLopt.ftol_abs!(opt, o.ftol_abs)
    NLopt.ftol_rel!(opt, o.ftol_rel)
    NLopt.maxeval!(opt, o.maxfeval)
    opt
end
