"""
    Layers.jl
# TODO: Module docstring
"""
module Layers

export NeuronLayer, SynapseLayer, LayeredNetwork, update!, propagate!, update_post!, runlayers!

include("./Neurons.jl")
include("./Synapses.jl")
include("./Utils.jl")

using .Neurons  
using .Synapses
using .Utils

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
    # TODO: add τ_m and R_m variation
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
    SynapseLayer
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
                         dist::AbstractDist, density::Float64)
        initw = clamp.(init_ws(dist, postlayer.N, prelayer.N), 0.01, template.wmax)
        bitmask = rand(postlayer.N, prelayer.N) .< density
        initw .*= bitmask

        return new(initw, template.wmax, template.learningrate, template.isinhibitory, template.delay)
    end
end


"""
    LayeredNetwork

A network composed of alternating neuron and synapse layers.

# Fields
- `neuronlayers::Vector{NeuronLayer}`: The neuron layers.
- `synapselayers::Vector{SynapseLayer}`: The synapse layers connecting neuron layers.
"""
struct LayeredNetwork
    neuronlayers::Vector{NeuronLayer}
    synapselayers::Vector{SynapseLayer}

    function LayeredNetwork(neurons::Vector{NeuronLayer}, synapses::Vector{SynapseLayer})
        length(synapses) == length(neurons) - 1 || throw(ArgumentError(
            "Expected $(length(neurons) - 1) synapse layers, got $(length(synapses))"))
        return new(neurons, synapses)
    end
end


"""
    runlayers!(network, dt, duration; t0=0.0, inputfn=nothing, callback=nothing) -> LayeredNetwork

Simulate the layered network for the given duration.

# Arguments
- `net::LayeredNetwork`: The network to simulate.
- `dt::Float64`: Time step.
- `duration::Float64`: Total simulation duration.
- `t0::Float64=0.0`: Optional start time.
- `inputfn`: Optional input function `inputfn(t, layer_idx) -> Float64` applied to each neuron layer at each step.
- `callback`: Optional callback `callback(t, net, step)` called after each time step.

# Returns
- `LayeredNetwork`: The updated network state.
"""
function runlayers!(net::LayeredNetwork, dt::Float64, duration::Float64; t0::Float64=0.0, inputfn=nothing, callback=nothing)
    nsteps = Int(round(duration / dt))
    nlayers = length(net.neuronlayers)
    n_synlayers = length(net.synapselayers)

    # Pre-allocate fired array and trace snapshots
    fired = [falses(net.neuronlayers[i].N) for i in 1:nlayers]
    old_pretraces = [copy(net.neuronlayers[i].pretraces) for i in 1:nlayers]
    old_posttraces = [copy(net.neuronlayers[i].posttraces) for i in 1:nlayers]

    for step in 1:nsteps
        t = t0 + (step - 1) * dt

        # Apply external input
        if inputfn !== nothing
            for (idx, layer) in enumerate(net.neuronlayers)
                layer.is .+= inputfn(t, idx)
            end
        end

        # Save current traces before updating 
        for i in 1:nlayers
            copyto!(old_pretraces[i], net.neuronlayers[i].pretraces)
            copyto!(old_posttraces[i], net.neuronlayers[i].posttraces)
        end

        # Update all neuron states and detect spikes
        for (i, layer) in enumerate(net.neuronlayers)
            fired[i] = update!(layer, dt, t)
        end

        # Synaptic propagation and STDP learning 
        @inbounds for i in 1:n_synlayers
            post_idx = i + 1
            post_idx > nlayers && continue

            pre_fired = fired[i]
            post_layer = net.neuronlayers[post_idx]
            pre_layer = net.neuronlayers[i]
            syn = net.synapselayers[i]

            # Forward propagation
            if any(pre_fired)
                propagate!(post_layer, syn, pre_fired, old_posttraces[post_idx])
            end
            
            # STDP weight update
            if any(fired[post_idx])
                update_post!(syn, fired[post_idx], old_pretraces[i])
            end
        end

        if callback !== nothing
            callback(t, net, step)
        end
    end

    return net
end


"""
    update!(layer, dt, t) -> BitArray

Advance the state of a neuron layer by one time step.

# Arguments
- `layer::NeuronLayer`: The neuron layer.
- `dt::Float64`: Time step.
- `t::Float64`: Current simulation time.

# Returns
- `BitArray`: Boolean array indicating which neurons fired.
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
    propagate!(post, syn, fired, post_posttraces)

Propagate pre-synaptic spikes to the post-synaptic layer and apply STDP LTD.

# Arguments
- `post::NeuronLayer`: Post-synaptic neuron layer.
- `syn::SynapseLayer`: Synapses connecting pre to post.
- `fired::BitArray`: Boolean vector indicating which pre-synaptic neurons fired.
- `post_posttraces::Vector{Float64}`: Post-synaptic spike traces from the *previous* timestep (for STDP causality).
"""
function propagate!(post::NeuronLayer, syn::SynapseLayer, fired::BitArray, post_posttraces::Vector{Float64})
    any(fired) || return

    # Apply weights
    w_impact = syn.isinhibitory ? -syn.ws : syn.ws
    post.is .+= sum(w_impact[:, fired], dims=2)[:]

    # STDP LTD
    ltd = syn.learningrate .* (post_posttraces * fired') .* (syn.ws ./ syn.wmax)
    syn.ws .= max.(0.0, syn.ws .- ltd) 
end

"""
    update_post!(syn, postfired, pre_pretraces)

Update synaptic weights based on post-synaptic spikes and pre-synaptic traces (STDP LTP).

# Arguments
- `syn::SynapseLayer`: Synapses to update.
- `postfired::BitArray`: Boolean vector indicating which post-synaptic neurons fired.
- `pre_pretraces::Vector{Float64}`: Pre-synaptic spike traces from the *previous* timestep (for STDP causality).
"""
function update_post!(syn::SynapseLayer, postfired::BitArray, pre_pretraces::Vector{Float64})
    any(postfired) || return
    
    # STDP LTP
    ltp = syn.learningrate .* (postfired * pre_pretraces') .* (1.0 .- syn.ws ./ syn.wmax)
    syn.ws .= min.(syn.wmax, syn.ws .+ ltp)
end

end # module Layers