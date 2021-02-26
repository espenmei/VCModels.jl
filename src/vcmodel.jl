"""
`VCData` stores fixed input data of a variance component model
# Fields
- `y`: 'n' vector of responses
- `X`: 'n × p' matrix of covariates
- `r`: 'q' vector of 'n × n' correlation matrices
- `dims`: tuple with n = length of y, p = columns of X, q = length of r
"""
struct VCData{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    #r::Vector{<:AbstractMatrix{T}} # Doesnt allow Float32 for A
    r::Vector{<:AbstractMatrix} # Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse!?
    dims::NamedTuple{(:n, :p, :q), NTuple{3, Int}}
end

function VCData(y::Vector{T}, X::VecOrMat{T}, r::Vector{<:AbstractMatrix}) where T <:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    VCData(
    y,
    X,
    r,
    (n = size(X, 1), p = size(X, 2), q = length(r))
    )
end

"""
`VCModel` holds data, parameters and optimization info of a variance component model
# Fields
- `data`: VCData
- `θ`: vector of scalar variance component parameters
- `θ_lb`: vector of lower bounds on θ
- `Λ`: cholesky factorization of the model implied covariance matrix
- `μ`: vector of model implied means
- `H`: matrix with missing or twice inverse covariance matrix of θ
- `opt`: NLopt.Opt
- `reml`: boolean indicator for reml
"""
#struct VCModel{T<:AbstractFloat, F<:Function} <:StatsBase.StatisticalModel
struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::VCData{T} # Type!?
    θ::Vector{T}
    θ_lb::Vector{T}
    #Λ::Cholesky{T}
    Λ::Cholesky{T, Matrix{T}}
    μ::Vector{T}
    H::Array{Union{Missing, T}, 2}
  #  f::F
    opt::Opt
    reml::Bool
end

#function VCModel(d::VCData, θ_init::Vector{T},  θ_lb::Vector{T}, f::Function, reml::Bool = false) where T<:AbstractFloat
function VCModel(d::VCData, θ_init::Vector{T},  θ_lb::Vector{T}, reml::Bool = false) where T<:AbstractFloat
    # Create new opt object and set parameters, same defaults as MixedModels.jl
    opt = Opt(:LN_BOBYQA, length(θ_init))
    lower_bounds!(opt, θ_lb) # lower bounds
    ftol_rel!(opt, T(1.0e-12)) # relative criterion on objective
    ftol_abs!(opt, T(1.0e-8)) # absolute criterion on objective   
    xtol_rel!(opt, zero(T)) # relative criterion on parameter values
    xtol_abs!(opt, T(1.0e-10)) # absolute criterion on parameter values   
    maxeval!(opt, -1) # maximum number of function evaluations
    
    n = d.dims.n
    q = d.dims.q
    VCModel(
    d,
    θ_init,
    θ_lb,
    cholesky(zeros(T, n, n) + I),
    Vector{T}(undef, n),
    Array{Union{Missing, T}}(missing, q, q),
 #   f,
    opt,
    reml
    )
end

function VCModel(d::VCData, θ_lb::Vector{<:AbstractFloat},  reml::Bool = false)
    msse = sum(abs2, d.y - d.X * (d.X \ d.y)) / d.dims.n # Initial values
    q = d.dims.q
    VCModel(
    d,
    fill(msse / q, q),
    θ_lb,
  #  (θ::Vector{T}) -> θ, # Just set to identity
    reml
    )
end

f(m::StatisticalModel) = m.θ

function update!(m::VCModel, θ::Vector)
    updateμ!(updateΛ!(setθ!(m, θ)))
    m
end

function setθ!(m::VCModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::VCModel)    
    fill!(m.Λ.factors, zero(eltype(m.θ)))
    δ = f(m) #δ = m.f(m.θ)
    for i in 1:m.data.dims.q # tar litt tid
        mul!(m.Λ.factors, δ[i], m.data.r[i], 1, 1)
    end
    cholesky!(Symmetric(m.Λ.factors, :U)) # Update the cholesky factorization object (Tar mest tid)
    m
end

# Generalized least squares for β
# Pawitan p. 440 (X'Σ^-1X)β = X'Σ^-1y
function updateμ!(m::VCModel)
    X = m.data.X
    ΣinvX = m.Λ \ X # Σ^-1X
    mul!(m.μ, X, (X' * ΣinvX) \ (ΣinvX' * m.data.y))
    m
end

function dfresidual(m::VCModel)::Int
    n = m.data.dims.n
    m.reml ? n - m.data.dims.p : n
end

# http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/10-chol/chol.html#Multivariate-normal-density
# Weighted residual sums of squares
# (y - Xβ)'Σ^-1(y - Xβ)
# Same as y'Py in Lynch & Walsh
# Same as trace(Σ^-1 * (y - Xβ)(y - Xβ)')
function wrss(m::VCModel)
    r = m.data.y - m.μ
    dot(r, m.Λ \ r)
end

# Pawitan p. 441
function rml(m::VCModel) # logdet(X' * Σ^-1 * X)
    X = m.data.X
    logdet(X' * (m.Λ \ X))
end

# Negative twice normal log-likelihood
# Is the constant right for reml?
function objective(m::VCModel)
    val = log(2π) * dfresidual(m) + logdet(m.Λ) + wrss(m)
    m.reml ? val + rml(m) : val
end

function fit(::Type{VCModel}, f::FormulaTerm, df::DataFrame, r::Vector, reml::Bool = false)
    sch = schema(f, df)
    form = apply_schema(f, sch)
    y, X = modelcols(form, df)
    d = VCData(y, X, r)
    θ_lb = fill(0.0, length(r))
    m = VCModel(d, θ_lb, reml)
    fit!(m)
end

function fit!(m::VCModel)
    if m.opt.numevals > 0
        throw(ArgumentError("This model has already been fitted"))
    end
    function obj(θ::Vector, g)
        val = objective(update!(m, θ))
        println("objective: $val, θ: $θ")
        val
    end
    min_objective!(m.opt, obj)
    minf, minx, ret = optimize!(m.opt, m.θ)
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

# Lynch & Walsh p. 789 
function fisherinfo!(m::VCModel)
    if m.opt.numevals <= 0
        @warn("This model has not been fitted")
        return nothing
    end
    n = m.data.dims.n
    Rstore = Matrix{eltype(m.θ)}(undef, n, n)
    q = m.data.dims.q
    for i ∈ 1:q, j ∈ 1:i
        ldiv!(Rstore, m.Λ, m.data.r[i])
        if j == i
            #m.H[i,j] = dot(r1, r1)
            m.H[i,j] = sum(abs2, Rstore)
        else
            #r2 = m.Λ \ m.data.r[j]
            m.H[i,j] = m.H[j,i] = dot(Rstore, m.Λ \ m.data.r[j])
        end
    end
end

function hessian!(m::VCModel)
    if m.opt.numevals <= 0
        @warn("This model has not been fitted")
        return nothing
    end
    function obj(x::Vector)
        val = objective(updateμ!(updateΛ!(setθ!(m_tmp, x))))
        println("objective: $val, θ: $x")
        val
    end
    m_tmp = deepcopy(m) # Finitediff kødder med med m under vurdering, så lag en kopi av alt og la den kødde der
    #cache = FiniteDiff.HessianCache(m.θ)
    FiniteDiff.finite_difference_hessian!(m.H, obj, m.θ)
end


# Lynch & Walsh p. 789
# They give it for ml, not obj, so remove the -0.5.
# https://www.biorxiv.org/content/10.1101/211821v1.full.pdf
function gradient(m::VCModel)
    q = m.data.dims.q
    g = Vector{eltype(m.θ)}(undef, q)
    r = m.Λ \ (m.data.y - m.μ)
    for i ∈ 1:q        
        g[i] = tr(m.Λ \ m.data.r[i]) - dot(r, m.data.r[i] * r)
    end
    g
end

function vcov(m::VCModel)
    X = m.data.X
    inv(X' * (m.Λ \ X))
end

function vcovvc(m::VCModel)
    H = m.H
    any(ismissing.(H)) ? H : inv(0.5 * H)
end

function vcovvctr(m::VCModel)
    J = FiniteDiff.finite_difference_jacobian(m.f, m.θ)
    J, J * vcovvc(m) * J'
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
    fixef!(Vector{T}(undef, m.data.dims.p), m)
end

# Posterior means for u
function ranef!(w::Matrix{T}, m::VCModel{T}) where T
    δ = m.f(m.θ)
    r = m.Λ \ (m.data.y - m.μ) # Σ^-1(y - Xβ) 
    for i in 1:m.data.dims.q
        w[:, i] = δ[i] * m.data.r[i] * r
    end
    w
end

function ranef(m::VCModel{T}) where T
    w = Matrix{T}(undef, m.data.dims.n, m.data.dims.q)
    ranef!(w, m)
end

# Implements
# Base
function Base.show(io::IO, m::VCModel)
    if m.opt.numevals <= 0
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
    criterion = m1.reml == m2.reml
    fterms = issubset(m1.data.X, m2.data.X)
    rterms = issubset(m1.data.r, m2.data.r)
    if m1.reml == true && m2.reml == true
        fterms = m1.data.X == m2.data.X
    end
    criterion && fterms && rterms
end