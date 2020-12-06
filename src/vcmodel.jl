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
    X::Array{T, 2}
    R::Vector{<:AbstractArray{T, 2}} # Abstract because they can be of different types, Symmetric, Diagonal, maybe also sparse!
    dims::NamedTuple{(:n, :p, :nvcomp), NTuple{3, Int}}
end

function VCData(y::Vector{T}, X::Array{T, 2}, R::Vector{AbstractArray{T, 2}}) where T<:AbstractFloat 
  #  Rs = Vector{AbstractArray{T, 2}}()
   # for i in R # Tag all correlation matrices as Hermition. Note that the matrix may change if not.
    #    push!(Rs, Hermitian(i))
    #end
    #n = size(X, 1)
    #push!(Rs, Diagonal(ones(n)))
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
- `θ`: array of scalar variance component parameters
- `θ_lb`: array of lowerbounds on θ
- `vc`: VCMat
- `H`: hessian
- `tf`: transformation function applied to θ during optimization
- `optsum`: OptSummary
"""
struct VCModel{T<:AbstractFloat}
    data::VCData
    θ::Vector{T}
    θ_lb::Vector{T}
    vc::VCMat{T}
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
    Array{Union{Missing, T}}(missing, 2, 2),
    tf,
    OptSummary(θ_init, θ_lb)
    )
end

function VCModel(d::VCData, θ_lb::Vector{T}) where T<:Real # AbstractFloat
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
    for i in 1:m.data.dims.nvcomp
        mul!(m.vc.Σ, δ[i], m.data.R[i], 1.0, 1.0)
    end
    # Update the cholesky factorization object
    cholesky!(Symmetric(copyto!(m.vc.Λ.factors, m.vc.Σ), :U))
    #m.vc.Λ = cholesky!(Σ) # Dette tar litt tid, men ikke mye minne
    m
end

function vcov(m::VCModel)
    X = m.data.X
    inv(X' * (m.vc.Λ \ X))
end

function vcovvc(m::VCModel)
    if any(ismissing.(m.H))
        return m.H
    else
        return inv(0.5 * m.H)
    end
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

# Denne er billig
function fixef(m::VCModel)
    X = m.data.X
    Λ = m.vc.Λ
    X' * (Λ \ X) \ (X' * (Λ \ m.data.y)) # GLS inv(X'inv(Σ)X)X'inv(Σ)y
   #  XtΣinvX = BLAS.gemm('T', 'N', X, Λ \ X)
   #  XtΣinvy = BLAS.gemv('T', X, Λ \ y)
   #  inv(XtΣinvX) * XtΣinvy
end

function ranef(m::VCModel{T}) where T
    nvcomp = m.data.dims.nvcomp
    U = Matrix{T}(undef, m.data.dims.n, nvcomp) # Fix type
    R = m.data.R
    r = m.vc.Λ \ (m.data.y - m.data.X * VCModels.fixef(m)) # Σ^-1(y - Xβ) 
    for i in 1:length(R)
        U[:, i] = m.θ[i] * R[i] * r
    end
    U
end

# Denne er nå billig!
# http://hua-zhou.github.io/teaching/biostatm280-2019spring/slides/10-chol/chol.html#Multivariate-normal-density
function wrss(m::VCModel) # (y - Xβ)'Σ^-1(y - Xβ) - r'r / n = residual var
    #r = m.vc.Λ.L \ (m.data.y - m.data.X * fixef(m))
    #r ⋅ r #
    r = m.data.y - m.data.X * VCModels.fixef(m)
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
        val = objective(updateΛ!(setθ!(m, θ)))
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
    numsoo = Ryu.writefixed.([-0.5 * oo, oo], 4) # decimals to print
    println(io, " Variance component model fit by maximum likelihood")
    println(io, "logLik", "\t\t", "-2 logLik")
    println(io, numsoo[1], "\t", numsoo[2])
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
