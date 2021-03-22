
#- `reml`: boolean indicator for reml
#- `H`: matrix with missing or twice inverse covariance matrix of θ
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
    ret::Symbol
    reml::Bool
    H::Matrix{Union{Missing, T}}
    ∇::Vector{Union{Missing, T}}
end

function VCOpt(optimizer::Symbol, xinitial::Vector{T}, xlb::Vector{T}, reml::Bool = false) where {T<:AbstractFloat}
    q = length(xlb)
    VCOpt(
        optimizer,
        xlb,
        T(10^-10),
        T(0),
        T(10^-8),
        T(10^-12),
        0,
        -1,
        xinitial,
        copy(xinitial),
        T(0),
        T(0),
        :FAILURE,
        reml,
        Matrix{Union{Missing, T}}(missing, q, q),
        Vector{Union{Missing, T}}(missing, q)
    )
end

# lag dennee så den kan kjøres hver iterasjon
#function update!(o::VCOpt, xinitial::Vector, finitial::T) where T<:AbstractFloat
    #o.xinitial = x.initial
    #o.finitial = finitial
    #o.xfinal = o.xinitial
    #o.dfinal = o.dinitial
    #o
#end

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

function converged(o::VCOpt)
    conv = true    
    f_abs = abs(o.ffinal - o.finitial)
    x_abs = abs.(o.xfinal - o.xinitial)
    
    if f_abs < o.ftol_abs
        o.ret = :FTOL_REACHED
    elseif all(x_abs .< o.xtol_abs)
        o.ret = :XTOL_REACHED
    elseif o.maxfeval >= 0 && o.feval >= o.maxfeval
        o.ret = :MAXEVAL_REACHED
    else
        conv = false
    end
    conv
end

#function Base.show(io::IO, ::MIME"text/plain", o::VCOpt)
   # for i ∈ 1:fieldcount(VCOpt)
  #      name = fieldname(VCOpt, i)
 #       val = getfield(o, i)
#        println(io, "$name = $val")
#    end
#end
#Base.show(io::IO, o::VCOpt) = Base.show(io, MIME"text/plain"(), o)
function Base.show(io::IO, o::VCOpt)
    for i ∈ 1:fieldcount(VCOpt)
        name = fieldname(VCOpt, i)
        val = getfield(o, i)
        println(io, "$name = $val")
    end
end