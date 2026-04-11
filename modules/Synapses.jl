"""
    Synapses.jl

Definitions for the `Synapse`` type and its STDP functions.
"""
module Synapses

export Synapse, decay!, prespike!, postspike!

"""
    Synapse

A directed connection between 2 `Neuron` instances with STDP.

# Fields 
- `inidx::Int`: Index of source/input neuron.
- `outidx::Int`: Index of target/output neuron.
- `w::Float64`: Connection weight.
- `wmax::Float64`: Maximum possible weight.
- `learningrate::Float64`: Synaptic learning rate.
- `isinhibitory::Bool`: Whether the synapse is inhibitory.
- `delay::Float64`: Synaptic transmission delay in time units.
"""
mutable struct Synapse
    inidx::Int
    outidx::Int
    w::Float64
    wmax::Float64
    learningrate::Float64
    isinhibitory::Bool
    delay::Float64
  
    @doc"""
        Synapse(inidx, outidx; kwargs...) -> Synapse
    Create a new `Synapse` instance. Inner constructor for the `Synapse` struct.

    # Arguments
    - `inidx::Int`: Index of the input neuron.
    - `outidx::Int`: Index of the output neuron.
    - `w::Float64=0.5`: Current synaptic weight.
    - `wmax::Float64=1.0`: Maximum synaptic weight.
    - `learningrate::Float64=0.01`: STDP learning weight.
    - `isinhibitory::Bool=false`: Whether this synapse is inhibitory.
    - `delay::Float64=0.0`: Synaptic transmission delay.
    """
    function Synapse(
        inidx::Int, 
        outidx::Int; 
        w::Float64 = 0.5, 
        wmax::Float64 = 1.0, 
        learningrate::Float64 = 0.01, 
        isinhibitory::Bool = false,
        delay::Float64 = 0.0
    )
        new(inidx, outidx, w, wmax, 0.0, 0.0, learningrate, isinhibitory, delay)
    end
end


"""
    prespike!(synapse, post-trace)

Update synapse state when the pre-synaptic neuron fires.
Reduces weight based on post-synaptic trace (LTD).
"""
function prespike!(syn::Synapse, posttrace::Float64)
    syn.pretrace += 1.0
    # LTD: weight decreases if post-synaptic neuron fired recently
    syn.w -= syn.learningrate * posttrace * (syn.w / syn.wmax)
    syn.w = max(0.0, syn.w)
end

"""
    postspike!(synapse, pre-trace)

Update synapse state when the post-synaptic neuron fires.
Increases weight based on pre-synaptic trace (LTP).
"""
function postspike!(syn::Synapse, pretrace::Float64)
    syn.posttrace += 1.0
    # LTP: weight increases if pre-synaptic neuron fired recently
    syn.w += syn.learningrate * pretrace * (1.0 - syn.w / syn.wmax)
    syn.w = min(syn.w, syn.wmax)
end

end #module Synapses