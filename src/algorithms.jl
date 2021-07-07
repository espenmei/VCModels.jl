
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
    tmp_n_i = zeros(T, n)
    P = copy(tmp_nn_i) # V^-1 for ml, P for reml
    
    while true
        o.feval += 1
        # initial values takes the last values
        o.xinitial .= o.xfinal
        o.finitial = o.ffinal
        
        copyto!(P, inv(m.Λ)) # expensive
        mul!(tmp_n_i, P, m.data.y - m.μ) # cheap
        if isreml(m)
            P .= invV2P(P, m.data.X) # cheap
        end
        expectedinfo!(m, P, tmp_n_i, tmp_nn_i) # expensive
        o.xfinal .= o.xinitial - o.H \ o.∇
        update!(m, o.xfinal) # Check bounds before update
        o.ffinal = objective(m)
        showiter(m.opt)

        if converged(m.opt)
            break
        end
    end
    m
end

function expectedinfo!(m::VCModel)
    n, _, q = m.data.dims
    T = eltype(m.θ)
    L = inv(m.Λ)
    invVϵ = L * (m.data.y - m.μ)
    if isreml(m)
        L .= invV2P(L, m.data.X)
    end
    tmp_nn = zeros(T, n, n)
    expectedinfo!(m, L, invVϵ, tmp_nn)
end

# Lynch & Walsh p. 789, # Undersøk denne med flere vc
function expectedinfo!(m::VCModel, L::AbstractMatrix, invVϵ::Vector, tmp_nn_i::Matrix)
    for i ∈ 1:m.data.dims.q
        mul!(tmp_nn_i, L, m.data.r[i])
        for j ∈ 1:i
            if j == i
                m.opt.H[i,j] = sum(abs2, tmp_nn_i)
                m.opt.∇[i] = tr(tmp_nn_i) - dot(invVϵ, m.data.r[i] * invVϵ)
            else
                m.opt.H[i,j] = m.opt.H[j,i] = dot(tmp_nn_i, L * m.data.r[j])
            end
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
    gradient!(∇, Symmetric(L), invVϵ, tmp_n, m)
end

  # Lynch & Walsh p. 789
  # They give it for ml, not obj, so remove the -0.5.
  # https://www.biorxiv.org/content/10.1101/211821v1.full.pdf
  # ML tr(V^-1 * R) - (y - Xβ) * 'V^-1 * R * V^-1 * (y - Xβ)
  # REML tr(P * R) - (y - Xβ)' * V^-1 * R * V^-1 * (y - Xβ)
  # P = V^-1 - Q
  function gradient!(∇::Vector, L::AbstractMatrix, invVϵ::Vector, tmp_n_i::Vector, m::VCModel)
    for i ∈ 1:m.data.dims.q
        mul!(tmp_n_i, m.data.r[i], invVϵ)
        ∇[i] = dot(L, m.data.r[i]) - dot(invVϵ, tmp_n_i)
    end
    ∇
end

function hessian(m::VCModel)
    q = m.data.dims.q
    hessian!(Matrix{eltype(m.θ)}(undef, q, q), m)
end

function hessian!(H::Matrix, m::VCModel)
    function obj(x::Vector)
        val = objective(update!(m_tmp, x))
        #showiter(m_tmp.opt)
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
    #FiniteDiff.finite_difference_jacobian!(J, f, copy(m.θ))
    J
    
end
