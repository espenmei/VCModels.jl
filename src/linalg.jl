
# Only fill the upper triangle
function LinearAlgebra.mul!(C::Matrix{T}, a::T, B::Symmetric{T, Matrix{T}}) where {T<:AbstractFloat}
    for i ∈ 1:size(C, 2)
        for j ∈ 1:i
           C[j,i] += a * B[j,i]
       end
   end
   C # Symmetric(C, :U)
end

function LinearAlgebra.mul!(C::Matrix{T}, a::T, B::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    for i ∈ 1:size(C, 2)
        C[i,i] += a * B[i,i]
    end
    C
end

function LinearAlgebra.dot(A::Symmetric{T}, B::Diagonal{T, Vector{T}}) where {T<:AbstractFloat}
    dot(A, Symmetric(B))
end

function LinearAlgebra.dot(A::Diagonal{T, Vector{T}}, B::Symmetric{T}) where {T<:AbstractFloat}
    dot(Symmetric(A), B)
end