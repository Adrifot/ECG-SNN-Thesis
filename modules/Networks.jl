"""
    Network.jl

Module description. # TODO: Add module docstring.
"""
module Networks

export Network, resolve_index

include("Neurons.jl")
include("Synapses.jl")

using .Neurons
using .Synapses

# TODO: Add docstrings for Network
#   [] Struct docstring
#   [] Constructor docstring
mutable struct Network
    neurons::Vector{Neuron}
    synapses::Vector{Synapse}
    index::Dict{String, Int}
    spikelog::Vector{Spike}

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

# TODO: Add docstrings for network expanders
#   [] Neuron
#   [] Synapse - 1st method
#   [] Synapse - 2nd method
"""
docstring
"""
function addneuron!(net::Network, n::Neuron)
    !haskey(net.index, n.name) || throw(ArgumentError("Duplicate neuron name: $(n.name)."))
    push!(net.neurons, n)
    net.index[n.name] = length(net.neurons)
    return net
end

"""
docstring
"""
function addsynapse!(net::Network, src_id::Int, target_id::Int)
    1 ≤ src_id ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ target_id ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    synapse = Synapse(src_id, target_id)
    push!(net.synapses, synapse)
    return net
end

function addsynapse!(net::Network, syn::Synapse)  
    1 ≤ syn.inidx ≤ length(net.neurons) || throw(ArgumentError("Source neuron id out of bounds."))
    1 ≤ syn.outidx ≤ length(net.neurons) || throw(ArgumentError("Target neuron id out of bounds."))
    push!(net.synapses, syn)
end

end # module Network