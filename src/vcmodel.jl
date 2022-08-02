struct VCCache{T<:AbstractFloat} 
    invVX::Matrix{T} # n × p
    XtinvVX::Matrix{T} # p × p
    β::Vector{T} # p XtinvVy -> β
    ϵ::Vector{T} # n
end

function VCCache(n, p, T)
    VCCache(
        zeros(T, n, p),
        zeros(T, p, p),
        zeros(T, p),
        zeros(T, n)
        )
end
 
"""
`VCData` holds input data of a variance component model
# Fields
- `y`: 'n' vector of responses
- `X`: 'n × p' matrix of covariates
- `r`: 'q' vector of 'n × n' correlation matrices
- `dims`: tuple of n = dimension of y, p = columns of X, q = dimension of r
"""
struct VCData{T<:AbstractFloat} 
    y::Vector{T}
    X::Matrix{T}
    r::Vector{<:AbstractMatrix} # Vector{<:AbstractMatrix{T}} Doesn't allow Float32 for A. Now it is not typestable. Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse? Change to tuple
    dims::NamedTuple{(:n, :p, :q), NTuple{3, Int}}
end

function VCData(y::Vector{T}, X::VecOrMat{T}, r::Vector{<:AbstractMatrix}) where T<:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    n, p = size(X)
    dims = (n = n, p = p, q = length(r))
    VCData(y, X, r, dims)
end

"""
`VCModel` holds data, parameters and optimization info of a variance component model
# Fields
- `data`: VCData
- `θ`: 'q' vector of scalar variance component parameters
- `Λ`: cholesky factorization of the 'n × n' model implied covariance matrix
- `μ`: 'n' vector of model implied means
- `opt`: VCOpt with optimization info
"""
#struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
#    data::VCData{T} # Type!?
#    θ::Vector{T} #StaticArray
#    Λ::Cholesky{T, Matrix{T}}
#    μ::Vector{T}
#    opt::VCOpt{T}
#end

#struct VCModel{T<:AbstractFloat, F<:Function} <:StatsBase.StatisticalModel
struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::VCData{T} # Type!?
    θ::Vector{T} # StaticArray?
    Λ::Cholesky{T, Matrix{T}}
    μ::Vector{T}
    opt::VCOpt{T}
    invVX::Matrix{T}
    #f::F
end

function VCModel(d::VCData, θ::Vector{T},  θ_lb::Vector{T}, reml::Bool = false) where T<:AbstractFloat
    n, p, _ = d.dims
    m = VCModel(
        d,
        copy(θ), # Make a copy to avoid modifying input
        Cholesky(zeros(T, n, n), :U, 0), # cholesky(zeros(T, n, n) + I),
        zeros(T, n),
        VCOpt(:LN_BOBYQA, copy(θ), θ_lb, reml),
        zeros(T, n, p)
        )
    update!(m, m.θ) # Gjør et update her?
    m.opt.finitial = objective(m)
    copyto!(m.opt.xfinal, m.opt.xinitial)
    m.opt.ffinal = m.opt.finitial
    m
end

function VCModel(d::VCData, θ_lb::Vector{<:AbstractFloat}, reml::Bool = false)
    VCModel(
    d,
    initialvalues(d),
    θ_lb,
    #(θ::Vector{T}) -> θ, # identity
    reml
    )
end

function initialvalues(d::VCData)
    n, _, q = d.dims
    X, y = d.X, d.y
    β = X \ y
    msse = sum(abs2, y - X * β) / n
    fill(msse / q, q)
end

isreml(m::VCModel) = m.opt.reml

#transform(m::StatisticalModel) = m.θ
transform(m::VCModel) = transform(m.θ)
transform(θ::Vector) = θ
#transform!(δ::Vector, θ::Vector) = copyto!(δ, θ)

function update!(m::VCModel, θ::Vector)
    updateμ!(updateΛ!(setθ!(m, θ)))
    m
end

function update!(m::VCModel, θ::Vector, c::VCCache)
    updateμ!(updateΛ!(setθ!(m, θ)), c)
    m
end

function setθ!(m::VCModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::VCModel)
    δ = transform(m.θ)
    Λfac = m.Λ.factors 
    fill!(Λfac, zero(eltype(δ)))
    @inbounds for i ∈ 1:m.data.dims.q
        muladduppertri!(Λfac, δ[i], m.data.r[i]) #mul!(m.Λ.factors, δ[i], m.data.r[i], 1, 1)
    end
    cholesky!(Symmetric(Λfac, :U), check = false) # Update the cholesky factorization object (Tar mest tid)
    # Can be harder to debug withut check
    m
end

#function updateΛp!(m::VCModel)    
#    δ = transform(m)
#    copyto!(m.Λ.factors, I)
#    for i ∈ 1:m.data.dims.q
#        scaleUpperTri!(m.Λ.factors, δ[i], m.data.r[i])
#    end
#    cholesky!(Symmetric(m.Λ.factors, :U))
#    m
#end

# Generalized least squares for β
# Pawitan p. 440 (X'V⁻¹X)β = X'V⁻¹y
# Allocates
function updateμ!(m::VCModel)
    X = m.data.X
    invVX = m.invVX
    ldiv!(invVX, m.Λ, X)
    #mul!(m.μ, X, ldiv!(cholesky!(Symmetric(X' * invVX)), (invVX' * m.data.y))) # Faster for larger p, but for some reason optim uses more itertions
    #mul!(m.μ, X, Symmetric(X' * invVX) \ (invVX' * m.data.y))
    # P/H = X, (X' * invVX) \ (invVX') -> "Hat matrix"
    mul!(m.μ, X, (X' * invVX) \ (invVX' * m.data.y))
    m
end

function updateμ!(m::VCModel, c::VCCache)
    X = m.data.X
    invVX = c.invVX
    println(@allocated ldiv!(invVX, m.Λ, X))
    println(@allocated mul!(c.XtinvVX, X', invVX))
    println(@allocated mul!(c.β, invVX', m.data.y))
    println(@allocated c.β .= Symmetric(c.XtinvVX) \ c.β)
    #ldiv!(factorize(Symmetric(c.XtinvVX)), c.β)
    println(@allocated mul!(m.μ, X, c.β))
    m
end

function dfresidual(m::VCModel)::Int
    n, p, _ = m.data.dims
    isreml(m) ? n - p : n
end

# http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/10-chol/chol.html#Multivariate-normal-density
# Weighted residual sums of squares
# (y - Xβ)'V^-1(y - Xβ)
# Same as y'Py in Lynch & Walsh
# Same as trace(V^-1 * (y - Xβ)(y - Xβ)')
function wrss(m::VCModel)
    ϵ = m.data.y - m.μ # Allocates
    dot(ϵ, m.Λ \ ϵ)
end

# Adjustment for reml likelihood
# Pawitan p. 441
# X' * V^-1 * X - It's computed in μ but cheap 
function rml(m::VCModel)
    X = m.data.X
    #logdet(X' * (m.Λ \ X))
    X' * m.invVX
end

# -2 × log-likelihood
function objective(m::VCModel)
    val = log(2π) * dfresidual(m) + logabsdet(m) + wrss(m)
    isreml(m) ? val + logdet(rml(m)) : val
end

#function objectivep(m::VCModel)
#    n = m.data.dims.n
#    logdet(m.Λ) + n * (1.0 + log(2π) + log(wrss(m) / n))
#end

# covariance of fixed effects
# Same as reml so make one X' * (m.Λ \ X) function 
function vcov(m::VCModel)
    X = m.data.X
    inv(X' * (m.Λ \ X))
end

# covariance of variance components
# scale to minimum of -2L
function vcovvc(m::VCModel)
    H = m.opt.H
    any(ismissing.(H)) ? H : inv(0.5 * H)
end

function stderror(m::VCModel)
    sqrt.(diag(vcov(m)))
end

function stderrorvc(m::VCModel)
    sqrt.(diag(vcovvc(m)))
end

function fixef!(v::Vector{T}, m::VCModel{T}) where T
    copyto!(v, m.data.X \ m.μ)
    v
end

function fixef(m::VCModel{T}) where T
    fixef!(zeros(T, m.data.dims.p), m)
end

# Posterior means for u
function ranef!(w::Matrix{T}, m::VCModel{T}) where T
    δ = transform(m)
    invVϵ = m.Λ \ (m.data.y - m.μ) # V^-1(y - Xβ) 
    for i in 1:m.data.dims.q
        w[:, i] = δ[i] * m.data.r[i] * invVϵ
    end
    w
end

function ranef(m::VCModel{T}) where T
    w = zeros(T, m.data.dims.n, m.data.dims.q)
    ranef!(w, m)
end

function fit(::Type{VCModel}, f::FormulaTerm, df::DataFrame, r::Vector, reml::Bool=false)
    sch = schema(f, df)
    form = apply_schema(f, sch)
    y, X = modelcols(form, df)
    d = VCData(y, X, r)
    θ_lb = fill(0.0, length(r))
    m = VCModel(d, θ_lb, reml)
    fit!(m)
end

function fit!(m::VCModel)
    if m.opt.feval > 0
        throw(ArgumentError("This model has already been fitted"))
    end
    # Det er jo egentlig gjort ett update når modellen ble laget. Men da må du stole på at modellen ikke har blitt klussa med.
    function obj(θ::Vector, g)
        val = objective(update!(m, θ))
        #val = objective(update!(m, θ, c))
        update!(m.opt, θ, val)
        showiter(m.opt)
        val
    end
    n, p = size(m.data.X)
    #c = VCCache(n, p, eltype(m.data.X)) #
    opt = Opt(m.opt)
    min_objective!(opt, obj)
    minf, minx, ret = optimize!(opt, m.θ)
    m.opt.ret = ret
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

# Implements
# StatsBase
StatsBase.coef(m::VCModel) = fixef(m)

function StatsBase.coeftable(m::VCModel)
    co = fixef(m)
    se = stderror(m)
    z = co ./ se
    pval = ccdf.(Chisq(1), abs2.(z))
    names = "x".*string.(1:length(co))
    tab = hcat(co, se, z, pval)
    CoefTable(
    tab, # value cols
    ["Coef.", "Std. Error", "z", "Pr(>|z|)"], # Colnames
    names, # rownames
    4, # pvalcol
    3,  # zcol
    )
end

StatsBase.deviance(m::VCModel) = objective(m)

StatsBase.dof(m::VCModel) = m.data.dims.p + m.data.dims.q

function StatsBase.dof_residual(m::VCModel)::Int
    m.data.dims.n - m.data.dims.p - m.data.dims.q
end

# Error for reml?
StatsBase.loglikelihood(m::VCModel) = -0.5 * objective(m)

StatsBase.modelmatrix(m::VCModel) = m.data.X

StatsBase.nobs(m::VCModel) = m.data.dims.n

StatsBase.response(m::VCModel) = m.data.y

# StatsModels
# It is diffiult to check rterms when constraints are imposed by manipulationg any R structure
function StatsModels.isnested(m1::VCModel, m2::VCModel; atol::Real = 0.0)
    response = m1.data.y == m2.data.y
    criterion = m1.opt.reml == m2.opt.reml
    #rterms = issubset(m1.data.r, m2.data.r)
    fterms = issubset(m1.data.X, m2.data.X)
    if m1.opt.reml && m2.opt.reml
        fterms = m1.data.X == m2.data.X
    end
    response && criterion && fterms
end

# Base
function Base.show(io::IO, m::VCModel)
    if m.opt.feval <= 0
        @warn("This model has not been fitted.")
        return nothing
    end
    # Fit measures
    oo = objective(m)
    oovals = Ryu.writefixed.([-0.5 * oo, oo, aic(m), aicc(m), bic(m)], 4)
    fieldwd = max(maximum(textwidth.(oovals)) + 1, 11)
    for i ∈ ["logLik", "-2 logLik", "AIC", "AICc", "BIC"]
        print(io, rpad(i, fieldwd))
    end
    println(io)
    for i ∈ oovals
        print(io, rpad(i, fieldwd))
    end
    # Variance components
    println(io, "\n\n Variance component parameters:")
    vcvals = Ryu.writefixed.(m.θ, 4)
    vcse = stderrorvc(m)
    vcsevals = any(ismissing.(vcse)) ? fill('-', length(vcse)) : Ryu.writefixed.(vcse, 4)
    for i ∈ ["Comp.", "Est.", "Std. Error"]
        print(io, i, "\t")
    end
    println(io)
    for i ∈ 1:length(vcvals)
        print(io, "θ" * Char(0x2080 + i), "\t", vcvals[i], "\t", vcsevals[i], "\n")
    end
    # Fixed effects
    println(io, "\n Fixed-effects parameters:")
    show(io, coeftable(m))
end