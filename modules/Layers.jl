"""
    Layers.jl
# TODO: Module docstring
"""
module Layers

export NeuronLayer, SynapseLayer, update!, propagate!, update_post!

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
    τ_pretrace::Float64
    τ_posttrace::Float64
    isreverse::Bool
    vs::Vector{Float64}
    is::Vector{Float64}
    t_refs::Vector{Float64}
    pretraces::Vector{Float64}
    posttraces::Vector{Float64}
    t_lastins::Vector{Float64}
    t_lastouts::Vector{Float64}

    @doc"""
    # TODO: docstring
    """
    function NeuronLayer(N::Int, template::Neuron; name::String="Layer")
        return new(
            N, name, 
            template.V_rest, template.V_thresh, template.V_reset, 
            template.R_m, template.τ_m, template.τ_s, template.τ_ref,
            template.τ_pretrace, template.τ_posttrace,
            template.isreverse, 
            fill(template.V_rest, N), 
            zeros(N), zeros(N), zeros(N), zeros(N), zeros(N), zeros(N)                  
        )
    end
end

"""
# TODO: Docstring
"""
struct SynapseLayer
    ws::Matrix{Float64}
    wmax::Float64
    learningrate::Float64
    isinhibitory::Bool
    delay::Float64

    @doc"""
        # TODO: Docstring
    """
    function SynapseLayer(prelayer::NeuronLayer, postlayer::NeuronLayer, template::Synapse;
                         randomweights::Bool=true, weightscale::Float64=0.25)
        initw = randomweights ? 
                rand(postlayer.N, prelayer.N) .* (template.wmax * weightscale) : 
                fill(template.w, postlayer.N, prelayer.N)
        
        return new(initw, template.wmax, template.learningrate, template.isinhibitory, template.delay)
    end
end


"""
# TODO: Docstring
"""
function update!(layer::NeuronLayer, dt::Float64, t::Float64)
    # Decay phase
    layer.is .*= exp(-dt / layer.τ_s)
    layer.pretraces .*= exp(-dt / layer.τ_pretrace)
    layer.posttraces .*= exp(-dt / layer.τ_posttrace)

    # LIF dynamics
    @. layer.vs += (-(layer.vs - layer.V_rest) + layer.R_m * layer.is) / layer.τ_m * dt
    
    # Refractory handling
    refmask = layer.t_refs .> 0
    layer.vs[refmask] .= layer.V_reset
    @. layer.t_refs = max(0.0, layer.t_refs - dt)

    # Spiking
    fired = layer.vs .>= layer.V_thresh

    if any(fired)
        layer.vs[fired] .= layer.V_reset 
        layer.t_refs[fired] .= layer.τ_ref
        layer.pretraces[fired] .+= 1.0
        layer.posttraces[fired] .+= 1.0
        layer.t_lastouts[fired] .= t
    end

    return fired
end

"""
# TODO: Docstring
"""
function propagate!(post::NeuronLayer, syn::SynapseLayer, fired::BitArray)
    any(fired) || return

    # Apply weights
    w_impact = syn.isinhibitory ? -syn.ws : syn.ws
    post.is .+= sum(w_impact[:, fired], dims=2)[:]

    # STDP LTD
    ltd = syn.learningrate .* (post.posttraces * fired') .* (syn.ws ./ syn.wmax)
    syn.ws .= max.(0.0, syn.ws .- ltd) 
end

"""
# TODO: Docstring
"""
function update_post!(pre::NeuronLayer, syn::SynapseLayer, postfired::BitArray)
    any(postfired) || return
    
    # STDP LTP
    ltp = syn.learningrate .* (postfired * pre.pretraces') .* (1.0 .- syn.ws ./ syn.wmax)
    syn.ws .= min.(syn.wmax, syn.ws .+ ltp)
end

end # module Layers