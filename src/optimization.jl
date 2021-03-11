
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
        showiter(val, θ)
        val
    end
    min_objective!(m.opt, obj)
    minf, minx, ret = optimize!(m.opt, m.θ)
    if ret ∈ [:FAILURE, :INVALID_ARGS, :OUT_OF_MEMORY, :FORCED_STOP, :MAXEVAL_REACHED]
        @warn("NLopt optimization failure: $ret")
    end
    m
end

# bounds
# reml
# transform, gradient? use Finitediff.joacobian? Or is the transformation always ortogonal?
function fs!(m::VCModel, maxiter = 100)

    if any(m.θ .!= transform(m))
        throw(ArgumentError("Transformatons of parameters are not allowed"))
    end

    if m.reml
        throw(ArgumentError("Fisher scoring is only available for models with reml = false"))
    end

    T = eltype(m.θ)
    n, p, q = m.data.dims
    F = zeros(T, q, q)
    ∇ = zeros(T, q)
    Rstore = zeros(T, n, n)
    θ = copy(m.θ)
    fval = objective(m)

    update!(m, m.θ)    
    
    for iter ∈ 1:maxiter
        invV = inv(m.Λ)
        invVr = m.Λ \ (m.data.y - m.μ)
        for i ∈ 1:q, j ∈ 1:i
            #ldiv!(Rstore, m.Λ, m.data.r[i])
            mul!(Rstore, invV, m.data.r[i])
            if j == i
                #g[i] = tr(Rstore) - dot(r, m.data.r[i] * r)
                #g[i] = dot(invV, m.data.r[i]) - dot(invVr, m.data.r[i] * invVr)
                ∇[i] = tr(Rstore) - dot(invVr, m.data.r[i] * invVr)
                F[i,j] = sum(abs2, Rstore)
            else
                #F[i,j] = F[j,i] = dot(Rstore, m.Λ \ m.data.r[j])
                F[i,j] = F[j,i] = dot(Rstore, invV * m.data.r[j])
                #Rstore .*= invV * m.data.r[j]
                #F[i,j] = F[j,i] = tr(Rstore)
            end
        end

        showiter(iter, m.θ, ∇)

        newθ = m.θ - inv(F) * ∇
        update!(m, newθ)
        newfval = objective(m)

        f_abs = fval - newfval
        if f_abs < 0.001
            println("absolute function tolerance reached")
            break
        end

        θ_abs = abs.(newθ - θ)
        #θ_rel = θ_abs ./ min(newθ, θ)
        if all(θ_abs .< 0.0005)
            println("absolute x-tolerance reached")
            break
        end
        θ = newθ
    end
    m
end

function em!(m::VCModel, iter = 100)
    n = m.data.dims.n
    update!(m, m.θ)
    θ = copy(m.θ)
    for i ∈ 1:iter
        g = VCModels.gradient(m)
        showiter(i, θ, g)
        θ .-= (θ.^2 ./ n) .* g
        update!(m, θ)
    end
    m
end

# Undersøk denne med flere vc!
# Sjekk hva som blir rett for reml
# Lynch & Walsh p. 789 
function fisherinfo!(m::VCModel)
    if any(m.θ .!= transform(m))
        throw(ArgumentError("Transformatons of parameters are not supported"))
    end
      if m.opt.numevals <= 0
          @warn("This model has not been fitted")
          return nothing
      end
    
      n, _, q = m.data.dims
      tmp_nn = Matrix{eltype(m.θ)}(undef, n, n)
      invV = inv(m.Λ)
      if m.reml
        X = m.data.X
        invVX = invV * X
        invV .-= (invVX / (X' * invVX))  * invVX'
      end

      for i ∈ 1:q, j ∈ 1:i
          #ldiv!(tmp_nn, m.Λ, m.data.r[i])
          mul!(tmp_nn, invV, m.data.r[i])
          if j == i
              m.H[i,j] = sum(abs2, tmp_nn)
          else
              #m.H[i,j] = m.H[j,i] = dot(tmp_nn, m.Λ \ m.data.r[j])
              m.H[i,j] = m.H[j,i] = dot(tmp_nn, invV * m.data.r[j])
          end
      end
  end

  # Parameter estimation and inference in the linear mixed model - F.N. Gumedze ∗, T.T. Dunne
  function observedinfo!(m::VCModel)    
    n, _, q = m.data.dims
    y = m.data.y
    T = eltype(m.θ)
    tmp_nn_i = Matrix{T}(undef, n, n)
    L = inv(m.Λ)
    invVr = L * (y - m.μ) # må defineres før if statement
    if m.reml
      X = m.data.X
      invVX = L * X
      L .-= (invVX / (X' * invVX))  * invVX'
    end
    for i ∈ 1:q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                m.H[i,j] = -sum(abs2, tmp_nn_i) + T(2) * invVr' * m.data.r[i] * tmp_nn_i * invVr
            else
                m.H[i,j] = m.H[j,i] = -dot(tmp_nn_i, L * m.data.r[j]) + T(2) * invVr' * tmp_nn_i * m.data.r[j] * invVr
            end
        end
    end
    m
  end

  function averageinfo!(m::VCModel)    
    n, _, q = m.data.dims
    y = m.data.y
    T = eltype(m.θ)
    tmp_nn_i = Matrix{T}(undef, n, n)
    L = inv(m.Λ)
    invVr = L * (y - m.μ) # må defineres før if statement
    if m.reml
      X = m.data.X
      invVX = L * X
      L .-= (invVX / (X' * invVX))  * invVX'
    end
    for i ∈ 1:q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                m.H[i,j] = invVr' * m.data.r[i] * tmp_nn_i * invVr
            else
                m.H[i,j] = invVr' * tmp_nn_i * m.data.r[j] * invVr
            end
        end
    end
    m
  end

  function hessian!(m::VCModel)
      if m.opt.numevals <= 0
          @warn("This model has not been fitted")
          return nothing
      end
      function obj(x::Vector)
          val = objective(update!(m_tmp, x))
          showiter(val, x)
          val
      end
      m_tmp = deepcopy(m) # Finitediff kødder med med m under vurdering, så lag en kopi av alt og la den kødde der
      #cache = FiniteDiff.HessianCache(m.θ)
      FiniteDiff.finite_difference_hessian!(m.H, obj, m.θ)
      m
  end
  
  function gradient(m::VCModel)
    gradient!(Vector{eltype(m.θ)}(undef, m.data.dims.q), m)
  end

  # Lynch & Walsh p. 789
  # They give it for ml, not obj, so remove the -0.5.
  # Onyl for ml, not reml
  # https://www.biorxiv.org/content/10.1101/211821v1.full.pdf
  # ML tr(V^-1 * R) - (y - Xβ) * 'V^-1 * R * V^-1 * (y - Xβ)
  # REML tr(P * R) - (y - Xβ)' * V^-1 * R * V^-1 * (y - Xβ)
  # P = V^-1 - Q
  function gradient!(∇::Vector, m::VCModel)
      invV = inv(m.Λ)
      if m.reml
        invV .= invV2P(invV, m.data.X)
      end
      invVr = invV * (m.data.y - m.μ) #invVr = m.Λ \ (m.data.y - m.μ)
      for i ∈ 1:m.data.dims.q
          #g[i] = tr(m.Λ \ m.data.r[i]) - dot(invVr, m.data.r[i] * invVr)
          ∇[i] = dot(invV, m.data.r[i]) - dot(invVr, m.data.r[i] * invVr)
      end
      ∇
  end

function invV2P(invV::Matrix, X::Matrix)
    invVX = invV * X
    invV .-= (invVX / (X' * invVX))  * invVX'
    invV
end

showiter(i::Int, v::Real, θ::Vector, ∇::Vector) = println("iteration: $i, objective: $v, θ: $θ, ∇: $∇")
showiter(i::Int, θ::Vector, ∇::Vector) = println("iteration: $i, θ: $θ, ∇: $∇")
showiter(θ::Vector, ∇::Vector) = println("θ: $θ, ∇: $∇")
showiter(v::Real, θ::Vector) = println("objective: $v, θ: $θ")