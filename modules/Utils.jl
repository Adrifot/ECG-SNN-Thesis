"""
    Utils.jl
# TODO: Docstring
"""
module Utils
export AbstractDist, NormalDist, ConstantDist, UniformDist, init_ws

using Distributions
using Random

abstract type AbstractDist end

struct UniformDist <: AbstractDist
    a::Float64
    b::Float64
end

struct NormalDist <: AbstractDist
    μ::Float64
    σ::Float64
end

struct ConstantDist <: AbstractDist
    val::Float64
end

function init_ws(dist::ConstantDist, x::Int, y::Int)
    return fill(dist.val, x, y)
end

function init_ws(dist::UniformDist, x::Int, y::Int)
    return rand(Uniform(dist.a, dist.b), x, y)
end    

function init_ws(dist::NormalDist, x::Int, y::Int)
    return rand(Normal(dist.μ, dist.σ), x, y)
end

end # module Utils