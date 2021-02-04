"""
`VCData` stores fixed input data of a variance component model.
# Fields
- `y`: 'n × 1' vector of responses
- `X`: 'n × p' matrix of covariates
- `R`: vector of 'n × n' correlation matrices
- `dims`: tuple with length of y, columns of X, and length of R
"""
struct VCData{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    #R::Vector{<:AbstractMatrix{T}}
    R::Vector{<:AbstractMatrix} # Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse!?
    dims::NamedTuple{(:n, :p, :nvcomp), NTuple{3, Int}}
end

#function VCData(y::Vector{T}, X::VecOrMat{T}, R::Vector{<:AbstractMatrix{T}}) where T <:AbstractFloat
function VCData(y::Vector{T}, X::VecOrMat{T}, R::Vector{<:AbstractMatrix}) where T <:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    VCData(
    y,
    X,
    R,
    (n = size(X, 1), p = size(X, 2), nvcomp = length(R))
    )
end

"""
`VCModel` holds data, parameters and optimization info of a variance component model.
# Fields
- `data`: VCData
- `θ`: vector of scalar variance component parameters
- `θ_lb`: vector of lower bounds on θ
- `Λ`: cholesky object of the model implied covariance matrix
- `μ`: vector of model implied means
- `H`: Matrix with missing or hessian if θ if requested
- `tf`: transformation function applied to θ during optimization
- `opt`: NLopt.Opt
"""
struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::VCData
    θ::Vector{T}
    θ_lb::Vector{T}
    Λ::Cholesky{T}  
    μ::Vector{T}
    H::Array{Union{Missing, T}, 2}
    tf::Function
    opt::Opt
    reml::Bool
end

function VCModel(d::VCData, θ_init::Vector{T},  θ_lb::Vector{T}, tf::Function, reml::Bool = false) where T<:AbstractFloat
    # Create new opt object and set parameters, same defaults as MixedModels.jl
    opt = Opt(:LN_BOBYQA, length(θ_init))
    lower_bounds!(opt, θ_lb) # lower bounds
    ftol_rel!(opt, T(1.0e-12)) # relative criterion on objective
    ftol_abs!(opt, T(1.0e-8)) # absolute criterion on objective   
    xtol_rel!(opt, zero(T)) # relative criterion on parameter values
    xtol_abs!(opt, T(1.0e-10)) # absolute criterion on parameter values   
    maxeval!(opt, -1) # maxumum number of function evaluations
    
    n = d.dims.n
    s = d.dims.nvcomp
    VCModel(
    d,
    θ_init,
    θ_lb,
    cholesky!(Matrix{T}(1.0I, n, n)),
    Vector{T}(undef, n),
    Array{Union{Missing, T}}(missing, s, s),
    tf,
    opt,
    reml
    )
end

function VCModel(d::VCData, θ_lb::Vector{T},  reml::Bool = false) where T<:AbstractFloat
    # Initial values
    msse = sum(abs2, d.y - d.X * (d.X \ d.y)) / d.dims.n
    s = d.dims.nvcomp
    VCModel(
    d,
    fill(msse / s, s),
    θ_lb,
    (θ::Vector{T}) -> θ, # Just set to identity
    reml
    )
end

function setθ!(m::VCModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::VCModel)    
    fill!(m.Λ.factors, zero(eltype(m.θ)))
    δ = m.tf(m.θ)
    for i in 1:m.data.dims.nvcomp # tar litt tid
        mul!(m.Λ.factors, δ[i], m.data.R[i], 1, 1)
        #BLAS.symm!('L', 'U', 1.0, m.data.R[i], δ[i], 1.0, m.Λ.factors)
        #axpy!(δ[i], m.data.R[i], m.Λ.factors)
    end
    # Update the cholesky factorization object
    cholesky!(Symmetric(m.Λ.factors, :U)) # Tar mest tid
    #cholesky!(Symmetric(copyto!(m.vc.Λ.factors, m.vc.Σ), :U)) #m.vc.Λ = cholesky!(Σ) # Dette tar litt tid, men ikke mye minne
    m
end

# Generalized lest squares for β
# Pawitan p. 440 (X'Σ^-1X)β = X'Σ^-1y
function updateμ!(m::VCModel)
    X = m.data.X
    ΣinvX = m.Λ \ X # Σ^-1X
    mul!(m.μ, X, (X' * ΣinvX) \ (ΣinvX' * m.data.y))
    m
end

function vcov(m::VCModel)
    X = m.data.X
    inv(X' * (m.Λ \ X))
end

function vcovvc(m::VCModel)
    H = m.H
    if !any(ismissing.(H))
        H = inv(0.5 * H)
    end
    H    
end

function vcovvctr(m::VCModel)
    J = FiniteDiff.finite_difference_jacobian(m.tf, m.θ)
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
    δ = m.tf(m.θ)
    r = m.Λ \ (m.data.y - m.μ) # Σ^-1(y - Xβ) 
    for i in 1:m.data.dims.nvcomp
        w[:, i] = δ[i] * m.data.R[i] * r
    end
    w
end

function ranef(m::VCModel{T}) where T
    w = Matrix{T}(undef, m.data.dims.n, m.data.dims.nvcomp)
    ranef!(w, m)
end

function dfresidual(m::VCModel)::Int
    n = m.data.dims.n
    m.reml ? n - m.data.dims.p : n
end

# Denne er nå billig!
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

# Are the constant right for reml?
# Negative twice normal log-likelihood
function objective(m::VCModel)
    val = log(2π) * dfresidual(m) + logdet(m.Λ) + wrss(m)
    m.reml ? val + rml(m) : val
    #log(2π) * m.data.dims.n + logdet(m.Λ) + wrss(m) #+ rml(m)
    #log(2π) * (m.data.dims.n - m.data.dims.p) + logdet(m.Λ) + wrss(m) + rml(m)
    #logdet(m.Λ) + wrss(m) + rml(m)
end

function fit(::Type{VCModel}, f::FormulaTerm, df::DataFrame, R::Vector, sevc::Bool = false, reml::Bool = false)
    sch = schema(f, df)
    form = apply_schema(f, sch)
    y, X = modelcols(form, df)
    d = VCData(y, X, R)
    θ_lb = fill(0.0, length(R))
    m = VCModel(d, θ_lb, reml)
    fit!(m, sevc)
end

function fit!(m::VCModel, sevc::Bool=false)
    if m.opt.numevals > 0
        throw(ArgumentError("This model has already been fitted."))
    end
    function obj(θ::Vector, g)
        val = objective(updateμ!(updateΛ!(setθ!(m, θ))))
        println("objective: $val, θ: $θ")
        val
    end
    min_objective!(m.opt, obj) # set obj as the function (to be minimized)
    #minf, minx, ret = optimize!(opt, copyto!(optsum.final, optsum.initial))
    minf, minx, ret = optimize!(m.opt, m.θ) # Optimize
    
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    
    if sevc
        @info("Computing hessian")
        hessian!(m)
    end
    m
end

function hessian!(m::VCModel)
    if m.opt.numevals <= 0
        @warn("This model has not been fitted.")
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
StatsBase.coef =(m::VCModel) = fixef(m)

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

StatsBase.dof(m::VCModel) = m.data.dims.p + m.data.dims.nvcomp

function StatsBase.dof_residual(m::VCModel)::Int
    m.data.dims.n - m.data.dims.p - m.data.dims.nvcomp
end

# Error for reml?
StatsBase.loglikelihood(m::VCModel) = -0.5 * objective(m)

StatsBase.modelmatrix(m::VCModel) = m.data.X

StatsBase.nobs(m::VCModel) = m.data.dims.n

StatsBase.response(m::VCModel) = m.data.y

# StatsModels
function StatsModels.isnested(m1::VCModel, m2::VCModel; atol::Real = 0.0)
    fterms = issubset(m1.data.X, m2.data.X)
    rterms = issubset(m1.data.R, m2.data.R)
    fterms && rterms
end