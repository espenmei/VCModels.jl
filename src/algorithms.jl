
# bounds
# transform, gradient? use Finitediff.jacobian?
function fs!(m::VCModel)
    if m.opt.feval > 0
        throw(ArgumentError("This model has already been fitted")) # Gjør dette til warning heller
    end
    if any(m.θ .!= transform(m))
        throw(ArgumentError("Transformations of parameters are not supported"))
    end
    T = eltype(m.θ)
    n = m.data.dims.n
    o = m.opt
    # tmp storage
    tmp_nn_i = zeros(T, n, n)
    tmp_nn_i2 = copy(tmp_nn_i)
    tmp_n_i = zeros(T, n)
    
    while true 
        o.feval += 1
        # initial values takes the last values
        copyto!(o.xinitial, o.xfinal)
        o.finitial = o.ffinal
        
        copyto!(tmp_nn_i2 ,inv(m.Λ)) # expensive
        mul!(tmp_n_i, tmp_nn_i2, m.data.y - m.μ) # cheap
        if isreml(m)
            tmp_nn_i2 .= invV2P(tmp_nn_i2, m.data.X) # cheap
        end
        expectedinfo!(o.H, o.∇, tmp_nn_i2, tmp_n_i, tmp_nn_i, m) # expensive
        # Check bounds before update
        copyto!(o.xfinal, o.xinitial - o.H \ o.∇)
        update!(m, o.xfinal)
        o.ffinal = objective(m)
        showiter(["iteration", "θ", "∇"], [o.feval, m.θ, o.∇])

        if converged(m.opt)
            break
        end
    end
    m
end

function em!(m::VCModel, iter=100)
    n, _, q = m.data.dims
    T = eltype(m.θ)
    θ = copy(m.θ)
    # tmp storage
    tmp_nn_i = zeros(T, n, n)
    tmp_n_i = zeros(T, n)

    for i ∈ 1:iter
        copyto!(tmp_nn_i ,inv(m.Λ))
        mul!(tmp_n_i, tmp_nn_i, m.data.y - m.μ)
        if isreml(m)
            tmp_nn_i .= invV2P(tmp_nn_i, m.data.X) # cheap
        end
        gradient!(m.opt.∇, tmp_nn_i, tmp_n_i, copy(tmp_n_i), m)
        showiter(i, θ, m.opt.∇)
        θ .-= (θ.^2 ./ n) .* m.opt.∇
        update!(m, θ)
    end
    m
end
  
function invV2P(invV::Matrix, X::Matrix)
    invVX = invV * X
    invV .-= (invVX / (X' * invVX))  * invVX'
    invV
end

function gradient(m::VCModel)
    n, _, q = m.data.dims
    T = eltype(m.θ)
    ∇ = zeros(T, q)
    L = inv(m.Λ)
    invVϵ = L * (m.data.y - m.μ) # = m.Λ \ (m.data.y - m.μ)
    if isreml(m)
        L .= invV2P(L, m.data.X)
    end
    tmp_n = zeros(T, n)
    gradient!(∇, L, invVϵ, tmp_n, m)
end

  # Lynch & Walsh p. 789
  # They give it for ml, not obj, so remove the -0.5.
  # https://www.biorxiv.org/content/10.1101/211821v1.full.pdf
  # ML tr(V^-1 * R) - (y - Xβ) * 'V^-1 * R * V^-1 * (y - Xβ)
  # REML tr(P * R) - (y - Xβ)' * V^-1 * R * V^-1 * (y - Xβ)
  # P = V^-1 - Q
  function gradient!(∇::Vector, L::Matrix, invVϵ::Vector, tmp_n_i::Vector, m::VCModel)
    for i ∈ 1:m.data.dims.q
        mul!(tmp_n_i, m.data.r[i], invVϵ)
        ∇[i] = dot(L, m.data.r[i]) - dot(invVϵ, tmp_n_i)
    end
    ∇
end

function averageinfo(m::VCModel)
    q = m.data.dims.q
    averageinfo!(Matrix{eltype(m.θ)}(undef, q, q), m)
end

function averageinfo!(H::Matrix, m::VCModel)    
    n, _, q = m.data.dims
    y = m.data.y
    tmp_nn_i = Matrix{eltype(m.θ)}(undef, n, n)
    L = inv(m.Λ)
    invVr = L * (y - m.μ) # må defineres før if statement
    if m.reml
      L .= invV2P(L, m.data.X)
    end
    for i ∈ 1:q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                H[i,j] = invVr' * tmp_nn_i * m.data.r[i] * invVr
            else
                H[i,j] = invVr' * tmp_nn_i * m.data.r[j] * invVr
            end
        end
    end
    Symmetric(H, :L)
end

function observedinfo(m::VCModel)
    q = m.data.dims.q
    observedinfo!(Matrix{eltype(m.θ)}(undef, q, q), m)
end

# Parameter estimation and inference in the linear mixed model - F.N. Gumedze ∗, T.T. Dunne
function observedinfo!(H::Matrix, m::VCModel)    
    n, _, q = m.data.dims
    y = m.data.y
    T = eltype(m.θ)
    tmp_nn_i = Matrix{T}(undef, n, n)
    L = inv(m.Λ)
    invVr = L * (y - m.μ) # må defineres før if statement
    if m.reml
        L .= invV2P(L, m.data.X)
    end
    for i ∈ 1:q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                H[i,j] = -sum(abs2, tmp_nn_i) + T(2) * invVr' * m.data.r[i] * tmp_nn_i * invVr
            else
                H[i,j] = -dot(tmp_nn_i, L * m.data.r[j]) + T(2) * invVr' * tmp_nn_i * m.data.r[j] * invVr
            end
        end
    end
    Symmetric(H, :L)
end

function expectedinfo(m::VCModel)
    n, _, q = m.data.dims
    T = eltype(m.θ)
    H = zeros(T, q, q)
    ∇ = zeros(T, q)
    L = inv(m.Λ)
    invVϵ = L * (m.data.y - m.μ)
    if isreml(m)
        L .= invV2P(L, m.data.X)
    end
    tmp_nn = zeros(T, n, n)
    expectedinfo!(H, ∇, L, invVϵ, tmp_nn, m)
end

# Lynch & Walsh p. 789, # Undersøk denne med flere vc
function expectedinfo!(H::Matrix, ∇::Vector, L::Matrix, invVϵ::Vector, tmp_nn_i::Matrix, m::VCModel)
    for i ∈ 1:m.data.dims.q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                H[i,j] = sum(abs2, tmp_nn_i)
                ∇[i] = tr(tmp_nn_i) - dot(invVϵ, m.data.r[i] * invVϵ)
            else
                H[i,j] = H[j,i] = dot(tmp_nn_i, L * m.data.r[j])
            end
        end
    end
    #Symmetric(H, :L), ∇
    H, ∇
end

function hessian(m::VCModel)
    q = m.data.dims.q
    hessian!(Matrix{eltype(m.θ)}(undef, q, q), m)
end

function hessian!(H::Matrix, m::VCModel)
    function obj(x::Vector)
        val = objective(update!(m_tmp, x))
        showiter(val, x)
        val
    end
    m_tmp = deepcopy(m) # Finitediff kødder med med m under vurdering, så lag en kopi av alt og la den kødde der
    FiniteDiff.finite_difference_hessian!(H, obj, copy(m.θ))
    H
end

function jacobian(m::VCModel)
    q = m.data.dims.q
    jacobian!(zeros(eltype(m.θ), q, q), m)
end

function jacobian!(J::Matrix, m::VCModel)
    function f(θ::Vector)
        update!(m_tmp, θ)
        transform(m_tmp)
    end
    m_tmp = deepcopy(m)
    J .= FiniteDiff.finite_difference_jacobian(f, copy(m.θ))
    J
end

function showvector(io, v::AbstractVector)
    print(io, "[")
    for (i, elt) in enumerate(v)
        i > 1 && print(io, ", ")
        print(io, elt)
    end
    print(io, "]")
end

function showiter(io, pre::AbstractVector, val::AbstractVector)
    p = length(pre)
    for i ∈ 1:p
        print(io, pre[i])
        print(io, ": ")
        if isa(val[i], Vector)
            showvector(io, val[i])
        else
            print(io, val[i])
        end
        i < p && print(io, ", ")
    end
    println(io)
end

showiter(p::AbstractVector, v::AbstractVector) = showiter(IOContext(stdout, :compact => true), p, v)

showiter(i::Int, v::Real, θ::Vector, ∇::Vector) = println("iteration: $i, objective: $v, θ: $θ, ∇: $∇")
showiter(i::Int, θ::Vector, ∇::Vector) = println("iteration: $i, θ: $θ, ∇: $∇")
showiter(θ::Vector, ∇::Vector) = println("θ: $θ, ∇: $∇")
showiter(v::Real, θ::Vector) = println("objective: $v, θ: $θ")
