"""
    Networks.jl

Utilities for building and simulating LIF Spiking Neural Networks 
connected by synapses with STDP capabilities.
"""
module Networks

export Network, resolve_index, addneuron!, addsynapse!, step!, run!,
     get_incoming_syns, get_outgoing_syns

include("Neurons.jl")
include("Synapses.jl")

using .Neurons
using .Synapses

"""
    Network
A collection of interconnected neurons and synapses. 

# Fields
- `neurons::Vector{Neuron}`: the `Neuron` instances that form the network.
- `synapses::Vector{Synapse}`: the `Synapse` instances connecting the neurons.
- `index::Dict{String, Int}`: A dictionary mapping neuron names to their id number.
- `spikelog::Vector{Spike}`: A vector of spikes outputted by the network.
"""
mutable struct Network
    neurons::Vector{Neuron}
    synapses::Vector{Synapse}
    index::Dict{String, Int}
    spikelog::Vector{Spike}

    @doc"""
        Network(neurons, synapses) -> Network
        Network(connectome) -> Network
        Network() -> Network
    
    Create a new `Network` instance. 
    Inner constructor for the `Network` struct.
    """
    function Network(
        ns::Vector{Neuron}, 
        syns::Vector{Synapse}
    )
        index = Dict{String, Int}()
        for (i, neuron) in enumerate(ns)
            !haskey(index, neuron.name) || throw(ArgumentError("Duplicate neuron name: $(neuron.name)."))
            index[neuron.name] = i
        end
        return new(ns, syns, index, Spike[])
    end
end

Network() = Network(Neuron[], Synapse[])

"""
    resolve_index(network, id) -> Int

Validate and return the integer index of a neuron within the `Network`.

If `id` is a `String`, it looks up the index in the network's internal mapping. 
If `id` is an `Int`, it verifies the index is within the valid range of 1 to the 
total number of neurons.

# Throws
- An `ArgumentError` if a `String` ID is not found or if an `Int` index is out of bounds.
"""
function resolve_index(net::Network, id::String)
    haskey(net.index, id) || throw(ArgumentError("Neuron with id $id not found."))
    return net.index[id]
end

function resolve_index(net::Network, id::Int)
    1 <= id <= length(net.neurons) || throw(ArgumentError("Neuron index out of range: $id"))
    return id
end

"""
    addneuron!(network, neuron) -> Network

Add a `Neuron` to a `Network`.
Duplicate neuron names not accepted.

# Returns
- `Network`: the updated network.
"""
function addneuron!(net::Network, n::Neuron)
    !haskey(net.index, n.name) || throw(ArgumentError("Duplicate neuron name: $(n.name)."))
    push!(net.neurons, n)
    net.index[n.name] = length(net.neurons)
    return net
end

"""
    addsynapse!(network, synapse) -> Network
    addsynapse!(network, source_neuron_id, target_neuron_id; synapse_kwargs...) -> Network
    addsynapse!(network, source_neuron_name, target_neuron_name; synapse_kwargs) -> Network

Add a `Synapse` to a `Network`. If source and target neuron ids are provided,
a `Synapse` with default parameters will be constructed.

# Returns
- `Network`: the updated network.
"""
function addsynapse!(net::Network, src_id::Int, target_id::Int; kwargs...)
    1 ≤ src_id ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ target_id ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    push!(net.synapses, Synapse(src_id, target_id; kwargs...))
    return net
end

function addsynapse!(net::Network, syn::Synapse)  
    1 ≤ syn.inidx ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ syn.outidx ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    push!(net.synapses, syn)
    return net
end

function addsynapse!(net::Network, pre_name::String, post_name::String; kwargs...)
    pre_idx = resolve_index(net, pre_name)
    post_idx = resolve_index(net, post_name)
    syn = Synapse(pre_idx, post_idx; kwargs...)
    push!(net.synapses, syn)
    return net
end

"""
    step!(network, dt, t)
    step!(connectome, dt, t)
    
Advance the state of the network by one time step. 

# Behavior
# TODO: document behavior
"""
function step!(net::Network, dt::Float64, t::Float64)
    N = length(net.neurons)
    fired = falses(N)

    for i in 1:N
        fired[i] = update!(net.neurons[i], dt, t)
    end

    for syn in net.synapses
        pren = net.neurons[syn.inidx]
        postn = net.neurons[syn.outidx]

        if fired[syn.inidx]
            prespike!(syn, postn.posttrace)
            receive_spike!(postn, syn.isinhibitory ? -syn.w : syn.w)
        end

        if fired[syn.outidx]
            postspike!(syn, pren.pretrace)
        end
    end

    for i in findall(fired)
        push!(net.spikelog, Spike(t, true, net.neurons[i].name))
    end
end

"""
    run!(network, input, input_target, dt, duration; t0=0.0, callback=nothing) -> Vector{Spike}
    run!(connectome, input, input_target, dt, duration; t0=0.0, callback=nothing) -> Vector{Spike}

  Run the network for the given duration using time step `dt`.

  # Arguments
- `net::Network` / `c::Connectome`: the network/connectome to simulate.
- `input`: Function providing input spikes at time t.
- `input_target::Vector{Int}`: Indices of neurons to receive the input spikes.
- `dt::Float64`: simulation time step.
- `duration::Float64`: total duration to run.
- `t0::Float64=0.0`: optional start time for the simulation.
- `callback`: Optional function called after each step.

  # Returns
  - `Vector{Spike}`: the network's `spikelog` after the run.
"""
function run!(net::Network, input, input_target::Vector{Int}, dt::Float64, duration::Float64; 
                t0::Float64=0.0, callback=nothing
            )
    nsteps = Int(round(duration / dt))
    empty!(net.spikelog)
    for step in 1:nsteps
        t = t0 + (step - 1) * dt
        for n in input_target
            receive_spike!(net.neurons[n], input(t)) 
        end
        
        step!(net, dt, t)
        if callback !== nothing
            callback(t, net, step)
        end
    end

    return net.spikelog
end



    """
    get_outgoing_syns(network, neuron_index) -> Vector{Synapse}

Return all synapses that originate from the neuron with index `idx`.
    """
get_outgoing_syns(net::Network, idx::Int) = filter(s -> s.inidx == idx, net.synapses)


    """
    get_incoming_syns(network, neuron_index) -> Vector{Synapse}

Return all synapses that target the neuron with index `idx`.
    """
get_incoming_syns(net::Network, idx::Int) = filter(s -> s.outidx == idx, net.synapses)

end # module Networks