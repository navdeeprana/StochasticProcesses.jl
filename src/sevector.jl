struct SEVector{T}
    v::Vector{T}
end
SEVector(T) = SEVector{T}(zeros(T, 1))
SEVector() = SEVector(Float64)
Base.getindex(x::SEVector, i) = @inbounds x.v[1]
Base.setindex!(x::SEVector, b, i) = @inbounds x.v[1] = b
Base.lastindex(x::SEVector) = 1
