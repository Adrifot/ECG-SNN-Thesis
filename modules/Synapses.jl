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
- `pretrace::Float64`: Trace of last input spike.
- `posttrace::Float64`: Trace of last output spike.
- `τ_pre::Float64`: Time constant for pre-synaptic trace.
- `τ_post::Float64`: Time constant for post-synaptic trace.
- `learningrate::Float64`: Synaptic learning rate.
- `isinhibitory::Bool`: Whether the synapse is inhibitory.
"""
mutable struct Synapse
    inidx::Int
    outidx::Int
    w::Float64
    wmax::Float64
    pretrace::Float64
    posttrace::Float64
    τ_pre::Float64
    τ_post::Float64
    learningrate::Float64
    isinhibitory::Bool
    # TODO: Add inner constructor docstirng
    function Synapse(
        inidx::Int, 
        outidx::Int; 
        w::Float64 = 0.5, 
        wmax::Float64 = 1.0, 
        τ_pre::Float64 = 20.0, 
        τ_post::Float64 = 20.0, 
        learningrate::Float64 = 0.01, 
        isinhibitory::Bool = false
    )
        new(inidx, outidx, w, wmax, 0.0, 0.0, τ_pre, τ_post, learningrate, isinhibitory)
    end
end


"""
    decay!(syn, dt)

Exponentially decay the pre- and post-synaptic traces.

# Arguments:
- `syn::Synapse`: The synapse to be decayed.
- `dt::Float64`: Time step used.
"""
function decay!(syn::Synapse, dt::Float64)
    syn.pretrace *= exp(-dt / syn.τ_pre)
    syn.posttrace *= exp(-dt / syn.τ_post)
end

"""
    prespike!(synapse)

Update synapse state when the pre-synaptic neuron fires.
Reduces weight based on post-synaptic trace (LTD).
"""
function prespike!(syn::Synapse)
    syn.pretrace += 1.0
    # LTD: weight decreases if post-synaptic neuron fired recently
    syn.w -= syn.learningrate * syn.posttrace * (syn.w / syn.wmax)
    syn.w = max(0.0, syn.w)
end

"""
    postspike!(synapse)

Update synapse state when the post-synaptic neuron fires.
Increases weight based on pre-synaptic trace (LTP).
"""
function postspike!(syn::Synapse)
    syn.posttrace += 1.0
    # LTP: weight increases if pre-synaptic neuron fired recently
    syn.w += syn.learningrate * syn.pretrace * (1.0 - syn.w / syn.wmax)
    syn.w = min(syn.w, syn.wmax)
end

end #module Synapses