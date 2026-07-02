"""
    Utils.jl

Utility helpers for initializing and perturbing network parameters.
"""
module Utils
export AbstractDist, NormalDist, ConstantDist, UniformDist, init_ws, _expand_vector_param, _vector_with_noise

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

function init_ws(dist::ConstantDist, x::Int, y::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    return fill(dist.val, x, y)
end

function init_ws(dist::UniformDist, x::Int, y::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    return rand(rng, Uniform(dist.a, dist.b), x, y)
end    

function init_ws(dist::NormalDist, x::Int, y::Int, rng::AbstractRNG=Random.GLOBAL_RNG)
    return rand(rng, Normal(dist.μ, dist.σ), x, y)
end

function _expand_vector_param(value, N::Int)
    v = isa(value, AbstractVector) ? collect(value) : fill(value, N)
    length(v) == N || throw(ArgumentError("Expected vector of length $N, got $(length(v))"))
    return v
end

function _vector_with_noise(value, deviation::Float64, N::Int; rng::AbstractRNG=Random.GLOBAL_RNG, minval=-Inf)
    base = _expand_vector_param(value, N)
    if deviation == 0.0
        return base
    end
    noisy = base .+ randn(rng, N) .* deviation
    return minval == -Inf ? noisy : clamp.(noisy, minval, Inf)
end

end # module Utils