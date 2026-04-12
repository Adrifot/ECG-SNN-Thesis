"""
    Layers.jl
# TODO: Module docstring    
"""
module Layers

#export

include("./Neurons.jl")
include("./Synapses.jl")

using .Neurons
using .Synapses

"""
    NeuronLayer
# TODO: Docstring
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

    @doc"""
        # TODO: Inner constructor Docstring
    """
    function NeuronLayer(N::Int, template::Neuron; name::String="Layer")
        return new(N, name, template.V_rest, template.V_thresh,
            template.R_m, template.τ_m, template.τ_s, template.τ_ref,
            template.isreverse, fill(template.V_rest, N), zeros(N), zeros(N))
    end
end


"""
    SynapseLayer
# TODO: Docstring
"""
struct SynapseLayer
    w::Matrix{Float64}
    wmax::Float64
    τ_pre::Float64
    τ_post::Float64
    learningrate::Float64
    isinhibitory::Bool
    delay::Float64

    @doc """
        # TODO: Inner constructor docstring
    """
    function SynapseLayer(prelayer::NeuronLayer, postlayer::NeuronLayer, template::Synapse;
                         randomweights::Bool=true, weightscale::Float64=0.25)
        if randomweights
            initw = rand(postlayer.N, prelayer.N) .* (template.wmax * weightscale)
        else
            initw = fill(template.w, postlayer.N, prelayer.N)
        end
        return new(initw, template.wmax, template.τ_pre, template.τ_post, 
                template.learningrate, template.isinhibitory, template.delay)
    end
end



end # module Layers

