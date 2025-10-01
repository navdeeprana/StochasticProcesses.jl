# Single Element Vector.
# Essentially implements an array that only has one element.
# x = RepeatedVector()
# x[i] = x[1] for all i.

struct RepeatedVector{T}
    v::Vector{T}
end

RepeatedVector{T}(val::T) where {T} = RepeatedVector{T}([val])
RepeatedVector(val::T) where {T} = RepeatedVector{T}([val])
RepeatedVector() = RepeatedVector(0.0)

Base.getindex(x::RepeatedVector, i) = @inbounds x.v[1]
Base.setindex!(x::RepeatedVector{T}, val::T, i) where {T} = @inbounds x.v[1] = val
Base.push!(x::RepeatedVector{T}, val::T) where {T} = x.v[1] = val

Base.lastindex(x::RepeatedVector) = typemax(Int)
Base.length(x::RepeatedVector) = typemax(Int)
Base.size(x::RepeatedVector) = (typemax(Int),)

Base.show(io::IO, x::RepeatedVector{T}) where {T} =
    print(io, "RepeatedVector{$T}($(x.v[1])) (Repeats infinitely)")
