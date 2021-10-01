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
