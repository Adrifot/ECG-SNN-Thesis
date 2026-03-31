"""
    Networks.jl

Utilities for building and simulating LIF Spiking Neural Networks 
connected by synapses with STDP capabilities.
"""
module Networks

export Network, resolve_index, addneuron!, addsynapse!, step!, run!,
     get_incoming_syns, get_outgoing_syns,
     Connectome, net_from_connectome, connectome_from_net

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
    addsynapse!(network, source_neuron_id, target_neuron_id) -> Network

Add a `Synapse` to a `Network`. If source and target neuron ids are provided,
a `Synapse` with default parameters will be constructed.

# Returns
- `Network`: the updated network.
"""
function addsynapse!(net::Network, src_id::Int, target_id::Int)
    1 ≤ src_id ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ target_id ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    push!(net.synapses, Synapse(src_id, target_id))
    return net
end

function addsynapse!(net::Network, syn::Synapse)  
    1 ≤ syn.inidx ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ syn.outidx ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    push!(net.synapses, syn)
    return net
end


"""
    step!(network, dt, t)
    step!(connectome, dt, t)
    
Advance the state of the network by one time step. This function updates all neurons,
decays synapses, propagates spikes through the network, and logs any spikes that occurred.

# Arguments
- `net::Network`: The network to update.
- `dt::Float64`: Time step used.
- `t::Float64`: Current simulation time.

# Behavior
1. Update each neuron and record which ones fired in a temporary array.
2. Decay all synapses exponentially:
    - is used on a `Network`, will apply **Global decay**.
    - if used on a `Connectome`, will apply **Lazy decay** but will work faster.
3. For each synapse where the pre-synaptic neuron fired:
    - Apply prespike update (LTD: decrease weight based on post-synaptic trace)
    - Inject synaptic current into the post-synaptic neuron
4. For each synapse where the post-synaptic neuron fired:
    - Apply postspike update (LTP: increase weight based on pre-synaptic trace)
5. Append all spikes from neurons that fired during this time step to the network's spike log.
"""
function step!(net::Network, dt::Float64, t::Float64)
    fired = falses(length(net.neurons))
    
    # 1. Update each neuron and store which ones fired
    for i in eachindex(net.neurons)
        fired[i] = update!(net.neurons[i], dt, t)
    end

    # 2. Decay all synapses
    for syn in net.synapses
        decay!(syn, dt)
    end

    # 3. Popagate current from pre-synaptic neurons
    neurons = net.neurons # local copy for faster access
    for syn in net.synapses
        if fired[syn.inidx]
            prespike!(syn)
            receive_spike!(neurons[syn.outidx], syn.isinhibitory ? -syn.w : syn.w)
        end
        if fired[syn.outidx]
            postspike!(syn)
        end
    end

    # 4. Log spikes
    for i in findall(identity, fired) # retreive only indeces with true values
        push!(net.spikelog, Spike(t, true, net.neurons[i].name))
    end
end

function step!(c::Connectome, dt::Float64, t::Float64)
    # 1. Track which neurons fired
    fired_indices = Int[]
    for i in eachindex(c.neurons)
        if update!(c.neurons[i], dt, t)
            push!(fired_indices, i)
        end
    end

    # 2 & 3. Update Synapses and propagate current
    for i in fired_indices
        for syn in c.outgoing[i]
            decay!(syn, dt) 
            prespike!(syn)
            target = c.neurons[syn.outidx]
            amt = syn.isinhibitory ? -syn.w : syn.w
            receive_spike!(target, amt)
        end

        for syn in c.incoming[i]
            postspike!(syn)
        end
    end

    # 4. Log spikes
    for i in fired_indices
        push!(c.spikelog, Spike(t, true, c.neurons[i].name))
    end
end

"""
    run!(network, dt, duration; t0=0.0) -> Vector{Spike}
    run!(connectome, dt, duration; t0=0.0) -> Vector{Spike}

Run the network for the given duration using time step `dt`.

# Arguments
- `net::Network`: the network to simulate.
- `dt::Float64`: simulation time step.
- `duration::Float64`: total duration to run.

# Keyword Arguments
- `t0::Float64=0.0`: optional start time for the simulation.

# Returns
- `Vector{Spike}`: the network's `spikelog` after the run.
"""
function run!(net::Network, dt::Float64, duration::Float64; t0::Float64=0.0)
    nsteps = Int(round(duration / dt))
    empty!(net.spikelog)
    for step in 1:nsteps
        t = t0 + (step - 1) * dt
        step!(net, dt, t)
    end

    return net.spikelog
end

function run!(c::Connectome, dt::Float64, duration::Float64; t0::Float64=0.0)
    nsteps = Int(round(duration / dt))
    empty!(c.spikelog)
    for step in 1:nsteps
        t = t0 + (step - 1) * dt
        step!(c, dt, t)
    end

    return c.spikelog
end

    """
    get_outgoing_syns(network, neuron_index) -> Vector{Synapse}

Return all synapses that originate from the neuron with integer index `idx`.
    """
get_outgoing_syns(net::Network, idx::Int) = filter(s -> s.inidx == idx, net.synapses)


    """
    get_incoming_syns(network, neuron_index) -> Vector{Synapse}

Return all synapses that target the neuron with integer index `idx`.
    """
get_incoming_syns(net::Network, idx::Int) = filter(s -> s.outidx == idx, net.synapses)


"""
    Connectome
A representation of an SNN network optimized for fast synaptic access via adjacency lists.

# Fields
- `neurons::Vector{Neuron}`: the `Neuron` instances that form the network.
- `outgoing::Vector{Vector{Synapse}}`: outgoing[i] is a vector of synapses from neuron i.
- `incoming::Vector{Vector{Synapse}}`: incoming[i] is a vector of synapses to neuron i.
- `spikelog::Vector{Spike}`: A vector of spikes outputted by the network.
"""
struct Connectome
    neurons::Vector{Neuron}
    outgoing::Vector{Vector{Synapse}}
    incoming::Vector{Vector{Synapse}}
    spikelog::Vector{Spike}

    @doc"""
        Connectome(neurons, synapses) -> Connectome
        Connectome(network) -> Connectome

    Create a new `Connectome` instance from neurons and synapses.
    Inner constructor for the `Connectome` struct.

    # Arguments
    - `neurons::Vector{Neuron}`: Vector of Neuron instances
    - `synapses::Vector{Synapse}`: Vector of Synapse instances
    """
    function Connectome(
        ns::Vector{Neuron}, 
        syns::Vector{Synapse}
    )
        n_neurons = length(ns)
        outgoing = [Synapse[] for _ in 1:n_neurons]
        incoming = [Synapse[] for _ in 1:n_neurons]
        for syn in syns
            push!(outgoing[syn.inidx], syn)
            push!(incoming[syn.outidx], syn)
        end
        return new(ns, outgoing, incoming, Spike[])
    end
end

function Network(c::Connectome)
    all_synapses = reduce(vcat, c.outgoing) 
    return Network(c.neurons, all_synapses)
end

Connectome(net::Network) = Connectome(net.neurons, net.synapses)

end # module Networks