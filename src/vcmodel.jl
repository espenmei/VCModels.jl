"""
`VCData` stores the fixed input data of a variance component model.
# Fields
- `y`: one-dimensional 'n × 1' vector of responses
- `X`: two-dimensional 'n × p' matrix of covariates
- `R`: vector of 'n × n' correlation matrices
- `dims`: tuple with size of y, X, and R
"""
struct VCData{T<:AbstractFloat}
    y::Vector{T}
    X::Matrix{T}
    R::Vector{<:AbstractMatrix{T}} # Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse!
    dims::NamedTuple{(:n, :p, :nvcomp), NTuple{3, Int}}
end

function VCData(y::Vector{T}, X::VecOrMat{T}, R::Vector{<:AbstractMatrix{T}}) where T <:AbstractFloat
    X = reshape(X, :, size(X, 2)) # Make sure X is a matrix
    VCData(
    y,
    X,
    R,
    (n = size(X, 1), p = size(X, 2), nvcomp = length(R))
    )
end

"""
`VCMat` holds the model implied covariance matrix.
# Fields
- `Λ`: cholesky object of the model covariance matrix
- `Σ`: model covariance matrix
"""
struct VCMat{T<:AbstractFloat}
    Λ::Cholesky{T}
    Σ::Matrix{T}
end

"""
`VCModel` holds data, parameters and optimization info of a variance component model.
# Fields
- `data`: VCData
- `θ`: vector of scalar variance component parameters
- `θ_lb`: vector of lower bounds on θ
- `vc`: VCMat
- `μ`: vector of model implied means
- `H`: Matrix with missing or hessian if requested
- `tf`: transformation function applied to θ during optimization
- `optsum`: OptSummary
"""
struct VCModel{T<:AbstractFloat} <:StatsBase.StatisticalModel
    data::VCData
    θ::Vector{T}
    θ_lb::Vector{T}
    vc::VCMat{T}
    μ::Vector{T}
    H::Array{Union{Missing, T}, 2}
    tf::Function
    optsum::OptSummary
end

function VCModel(d::VCData, θ_lb::Vector{T}, tf::Function) where T<:AbstractFloat
    # Initial values
    msse = sum((d.y - d.X * (d.X \ d.y)).^2) / d.dims.n
    s = d.dims.nvcomp
    θ_init = fill(msse / s, s)
    n = d.dims.n
    VCModel(
    d,
    θ_init,
    θ_lb,
    VCMat(cholesky!(Matrix{T}(1.0I, n, n)), Matrix{T}(undef, n, n)),
    Vector{T}(undef, n),
    Array{Union{Missing, T}}(missing, s, s),
    tf,
    OptSummary(θ_init, θ_lb)
    )
end

function VCModel(d::VCData, θ_lb::Vector{T}) where T<: AbstractFloat
    VCModel(
    d,
    θ_lb,
    (θ::Vector{T}) -> θ # Just set to identity
    )
end

function setθ!(m::VCModel, θ::Vector)
    copyto!(m.θ, θ)
    m
end

function updateΛ!(m::VCModel)
    δ = m.tf(m.θ)
    #Σ = sum(broadcast(*, δ, m.data.R)) # Dette er dyrt
    fill!(m.vc.Σ, zero(eltype(m.θ))) # Reset all values in Σ to zero
    begin
    for i in 1:m.data.dims.nvcomp # tar litt tid
        mul!(m.vc.Σ, δ[i], m.data.R[i], 1.0, 1.0)
        #LinearAlgebra.axpy!(δ[i], m.data.R[i], m.vc.Σ)
    end
    end
    # Update the cholesky factorization object
    copyto!(m.vc.Λ.factors, m.vc.Σ)
    cholesky!(Symmetric(m.vc.Λ.factors, :U)) # Tar mest tid
    #cholesky!(Symmetric(copyto!(m.vc.Λ.factors, m.vc.Σ), :U)) #m.vc.Λ = cholesky!(Σ) # Dette tar litt tid, men ikke mye minne
    m
end

function updateμ!(m::VCModel)
    mul!(m.μ, m.data.X, fixef(m))
    m
end

function vcov(m::VCModel)
    X = m.data.X
    inv(X' * (m.vc.Λ \ X))
end

function vcovvc(m::VCModel)
    H = m.H
    if !any(ismissing.(m.H))
        H = inv(0.5 * m.H)
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
    X = m.data.X
    Λ = m.vc.Λ
    #copyto!(v, X' * (Λ \ X) \ (X' * (Λ \ m.data.y))) # Denne er billig
    ΣinvX = Λ \ X # Σ^-1X
    copyto!(v, (X' * ΣinvX) \ (ΣinvX' * m.data.y))
    v
end

function fixef(m::VCModel{T}) where T
    fixef!(Vector{T}(undef, m.data.dims.p), m)
end

function ranef!(w::Matrix{T}, m::VCModel{T}) where T
    δ = m.tf(m.θ)
    r = m.vc.Λ \ (m.data.y - m.μ) # Σ^-1(y - Xβ) 
    for i in 1:m.data.dims.nvcomp
        w[:, i] = δ[i] * m.data.R[i] * r
    end
    w
end

function ranef(m::VCModel{T}) where T
    w = Matrix{T}(undef, m.data.dims.n, m.data.dims.nvcomp)
    ranef!(w, m)
end

# Denne er nå billig!
# http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/10-chol/chol.html#Multivariate-normal-density
# Weighted residual sums of squares
function wrss(m::VCModel) # (y - Xβ)'Σ^-1(y - Xβ) = r'r
    #r = m.vc.Λ.L \ (m.data.y - m.data.X * fixef(m))
    #r ⋅ r #
    r = m.data.y - m.μ
    dot(r, m.vc.Λ \ r)
end

function objective(m::VCModel)    
    log(2.0π) * m.data.dims.n + logdet(m.vc.Λ) + wrss(m)
end

function fit(::Type{VCModel}, f::FormulaTerm, df::DataFrame, R::Vector, sevc::Bool=false)
    sch = schema(f, df)
    form = apply_schema(f, sch)
    y, X = modelcols(form, df)
    d = VCData(y, X, R)
    θ_lb = fill(0.0, length(R))
    m = VCModel(d, θ_lb)
    fit!(m, sevc)
end

function fit!(m::VCModel, sevc::Bool=false)
    optsum = m.optsum
    if optsum.feval > 0
        throw(ArgumentError("This model has already been fitted."))
    end
    opt = Opt(optsum)
    function obj(θ::Vector, dummygrad)
        val = objective(updateμ!(updateΛ!(setθ!(m, θ))))
        println("objective: $val, θ: $θ")
        val
    end
    NLopt.min_objective!(opt, obj)
    minf, minx, ret = NLopt.optimize!(opt, copyto!(optsum.final, optsum.initial))
    # Update optsum
    optsum.feval = opt.numevals
    optsum.fmin = minf
    optsum.returnvalue = ret
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    # Hessian
    if sevc
        @info("Computing hessian")
        hessian!(m)
    end
    m
end

function hessian!(m::VCModel)
    m_tmp = deepcopy(m)
    function obj(x::Vector)
        val = objective(updateΛ!(setθ!(m_tmp, x)))
        println("objective: $val, θ: $x")
        val
    end
    # Need to use opsum.final here as m.θ is not mutable?
    FiniteDiff.finite_difference_hessian!(m.H, obj, m_tmp.optsum.final)
    m
end

function Base.show(io::IO, m::VCModel)
    if m.optsum.feval < 0
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

function StatsBase.coeftable(m::VCModel)
    co = fixef(m)
    se = stderror(m)
    z = co ./ se
    pval = ccdf.(Chisq(1), abs2.(z))
    names = "x".*string.(1:length(co))
    tab = hcat(co, se, z, pval)
    CoefTable(
    tab, # value cols
    ["Coef.", "Std. Error", "z", "Pr(>|z|)"], # Colnms
    names, # rownames
    4, # pvalcol
    3,  # zcol
    )
end

function StatsBase.dof(m::VCModel)
    m.data.dims.p + m.data.dims.nvcomp
end

function StatsBase.nobs(m::VCModel)
    m.data.dims.n
end

function StatsBase.loglikelihood(m::VCModel)
    -0.5 * objective(m)
end