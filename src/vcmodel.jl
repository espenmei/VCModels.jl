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
    r::Vector{<:AbstractMatrix} # Vector{<:AbstractMatrix{T}} Doesn't allow Float32 for A # Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse?
    dims::NamedTuple{(:n, :p, :q), NTuple{3, Int}}
end

function VCData(y::Vector{T}, X::VecOrMat{T}, r::Vector{<:AbstractMatrix}) where T<:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    dims = (n = size(X, 1), p = size(X, 2), q = length(r))
    VCData(y, X, r, dims)
end

"""
`VCModel` holds data, parameters and optimization info of a variance component model
# Fields
- `data`: VCData
- `θ`: vector of scalar variance component parameters
- `Λ`: cholesky factorization of the model implied covariance matrix
- `μ`: vector of model implied means
- `opt`: VCOpt with optimization info
"""
struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::VCData{T} # Type!?
    θ::Vector{T}
    Λ::Cholesky{T, Matrix{T}}
    μ::Vector{T}
    opt::VCOpt{T}
end

function VCModel(d::VCData, θ::Vector{T},  θ_lb::Vector{T}, reml::Bool = false) where T<:AbstractFloat
    n = d.dims.n
    m = VCModel(
        d,
        θ,
        cholesky(zeros(T, n, n) + I),
        zeros(T, n),
        VCOpt(:LN_BOBYQA, copy(θ), θ_lb, reml)
        )
    # Gjør et update her?
    update!(m, m.θ)
    m.opt.finitial = objective(m)
    copyto!(m.opt.xfinal, m.opt.xinitial)
    m.opt.ffinal = m.opt.finitial
    m
end

function VCModel(d::VCData, θ_lb::Vector{<:AbstractFloat},  reml::Bool = false)
    VCModel(
    d,
    initialvalues(d),
    θ_lb,
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

transform(m::StatisticalModel) = m.θ

function update!(m::VCModel, θ::Vector)
    updateμ!(updateΛ!(setθ!(m, θ)))
    m
end

function setθ!(m::VCModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::VCModel)
    δ = transform(m)
    fill!(m.Λ.factors, zero(eltype(δ)))
    for i ∈ 1:m.data.dims.q
        muladduppertri!(m.Λ.factors, δ[i], m.data.r[i]) #mul!(m.Λ.factors, δ[i], m.data.r[i], 1, 1)
    end
    # Does the error for non-PD comes from cholesky?
    cholesky!(Symmetric(m.Λ.factors, :U)) # Update the cholesky factorization object (Tar mest tid)
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
# Pawitan p. 440 (X'V^-1X)β = X'V^-1y
function updateμ!(m::VCModel)
    X = m.data.X
    invVX = m.Λ \ X # V^-1X # allocations
    mul!(m.μ, X, (X' * invVX) \ (invVX' * m.data.y))
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
    ϵ = m.data.y - m.μ
    dot(ϵ, m.Λ \ ϵ)
end

# Adjustment for reml likelihood
# Pawitan p. 441
# X' * V^-1 * X - It's computed in μ but cheap 
function rml(m::VCModel)
    X = m.data.X
    logdet(X' * (m.Λ \ X))
end

# Negative twice normal log-likelihood
function objective(m::VCModel)
    #val = log(2π) * dfresidual(m) + logdet(m.Λ) + wrss(m)
    val = log(2π) * dfresidual(m) + logabsdet(m) + wrss(m)
    isreml(m) ? val + rml(m) : val
end

#function objectivep(m::VCModel)
#    n = m.data.dims.n
#    logdet(m.Λ) + n * (1.0 + log(2π) + log(wrss(m) / n))
#end


# covariance of fixed effects
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
        update!(m.opt, θ, val)
        showiter(m.opt)
        val
    end
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
# Check that both are reml or ml. For reml X == X must hold.
function StatsModels.isnested(m1::VCModel, m2::VCModel; atol::Real = 0.0)
    criterion = m1.opt.reml == m2.opt.reml
    fterms = issubset(m1.data.X, m2.data.X)
    rterms = issubset(m1.data.r, m2.data.r)
    if m1.opt.reml && m2.opt.reml
        fterms = m1.data.X == m2.data.X
    end
    criterion && fterms && rterms
end

# Base
function Base.show(io::IO, m::VCModel)
    if m.opt.feval <= 0
        @warn("This model has not been fitted.")
        return nothing
    end
    oo = objective(m)
    nums = Ryu.writefixed.([-0.5 * oo, oo, aic(m), aicc(m), bic(m)], 4)
    cols = ["logLik", "-2 logLik", "AIC", "AICc", "BIC"]
    fieldwd = max(maximum(textwidth.(nums)) + 1, 11)
    for i in cols
        print(io, rpad(i, fieldwd))
    end
    println(io)
    for i in nums
        print(io, rpad(i, fieldwd))
    end
    println(io)
    println(io)
    println(io, " Variance component parameters:")

    numsvc = Ryu.writefixed.(m.θ, 4)
    vcse = stderrorvc(m)
    numsvcse = fill('-', length(vcse))
    if !any(ismissing.(vcse))
        numsvcse = Ryu.writefixed.(vcse, 4)
    end
    for label in ["Comp.", "Est.", "Std. Error"]
        print(io, label, "\t")
    end
    println(io)
    for i in 1:length(numsvc)
        print(io, "θ" * Char(0x2080 + i), "\t", numsvc[i], "\t", numsvcse[i], "\n")
    end
    println(io)
    println(io, " Fixed-effects parameters:")
    show(io, coeftable(m))
end