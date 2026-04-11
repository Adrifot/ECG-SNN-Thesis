"""
    Layers.jl
Module docstring    
"""
module Layers

#export

include("./Neurons.jl")
include("./Synapses.jl")

using .Neurons
using .Synapses

"""
    NeuronLayer
Docstring
"""
struct NeuronLayer
    N::Int
    name::String
    V_rest::Float64
    V_thresh::Float64
    V_reset::Float64
    R_m::Float64
    τ_m::Float64
    τ_s::Float64
    τ_ref::Float64
    isreverse::Bool
    v::Vector{Float64}
    i::Vector{Float64}
    t_ref::Vector{Float64}
end


"""
    SynapseLayer
Docstring
"""
struct SynapseLayer
    w::Matrix{Float64}
    wmax::Float64
    τ_pre::Float64
    τ_post::Float64
    learningrate::Float64
    isinhibitory::Bool
    delay::Float64
end



end # module Layers

