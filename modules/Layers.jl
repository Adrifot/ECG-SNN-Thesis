"""
    Layers.jl

A module defining layered spiking neural network components and simulation routines.

# Exports:
- `NeuronLayer` struct: represents a layer of neurons built using a `Neuron` struct as a template with variations to some characteristic values.
- `SynapseLayer` struct: similar to `NeuronLayer`, but for Synapses. Supports recurrent connections and lateral inhibition.
- `update!`, `update_post!`, `propagate!`, `runlayers!` functions used for the STDP logic and simulation.
"""
module Layers

export NeuronLayer, SynapseLayer, LayeredNetwork, update!, propagate!, update_post!, runlayers!

include("./Neurons.jl")
include("./Synapses.jl")
include("./Utils.jl")

using .Neurons  
using .Synapses
using .Utils
using Random
using LinearAlgebra

"""
    NeuronLayer
# TODO: Docstring
"""
struct NeuronLayer
    N::Int
    name::String
    V_rest::Float64
    V_thresh::Vector{Float64}
    V_reset::Float64
    R_m::Vector{Float64}
    τ_m::Vector{Float64}
    τ_s::Float64
    τ_ref::Float64
    τ_pretrace::Float64
    τ_posttrace::Float64
    isreverse::Bool
    v::Vector{Float64}
    i::Vector{Float64}
    t_ref::Vector{Float64}
    pretraces::Vector{Float64}
    posttraces::Vector{Float64}
    t_lastin::Vector{Float64}
    t_lastout::Vector{Float64}

    @doc"""
    # TODO: docstring
    """
    function NeuronLayer(
        N::Int,
        template::Neuron;
        name::String="Layer",
        V_thresh_dev::Float64=0.0,
        R_m_dev::Float64=0.0,
        τ_m_dev::Float64=0.0,
        rng::AbstractRNG=Random.GLOBAL_RNG
    )
        V_thresh = _vector_with_noise(
            template.V_thresh,
            V_thresh_dev,
            N;
            rng=rng,
            minval=template.V_rest + eps()
        )
        R_m = _vector_with_noise(template.R_m, R_m_dev, N; rng=rng, minval=eps())
        τ_m = _vector_with_noise(template.τ_m, τ_m_dev, N; rng=rng, minval=eps())

        return new(
            N,
            name,
            template.V_rest,
            V_thresh,
            template.V_reset,
            R_m,
            τ_m,
            template.τ_s,
            template.τ_ref,
            template.τ_pretrace,
            template.τ_posttrace,
            template.isreverse,
            fill(template.V_rest, N),
            zeros(N),
            zeros(N),
            zeros(N),
            zeros(N),
            zeros(N),
            fill(-Inf, N)
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
    pre_idx::Int
    post_idx::Int

    @doc"""
        # TODO: Docstring
    """
    function SynapseLayer(prelayer::NeuronLayer, postlayer::NeuronLayer, template::Synapse;
                         dist::AbstractDist, density::Float64, pre_idx::Int, post_idx::Int,
                         rng::AbstractRNG=Random.GLOBAL_RNG)
        initw = clamp.(init_ws(dist, postlayer.N, prelayer.N, rng), 0.01, template.wmax)
        bitmask = rand(rng, postlayer.N, prelayer.N) .< density
        initw .*= bitmask

        # Zero diagonal for lateral inhibition to prevent self-inhibition
        if template.isinhibitory && prelayer === postlayer
            initw[diagind(initw)] .= 0
        end

        return new(initw, template.wmax, template.learningrate, template.isinhibitory, template.delay, pre_idx, post_idx)
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
end


"""
    runlayers!(network, dt, duration; t0=0.0, inputfn=nothing, callback=nothing) -> LayeredNetwork

Simulate the layered network for the given duration.

# Arguments
- `net::LayeredNetwork`: The network to simulate.
- `dt::Float64`: Time step.
- `duration::Float64`: Total simulation duration.
- `t0::Float64=0.0`: Optional start time.
- `inputfn`: Optional input function `inputfn(t) -> Float64` applied to each neuron in the input layer at each step.
 - `callback`: Optional callback `callback(t, net, step)` called after each time step.
 - `freeze_at::Float64=Inf`: Time after which STDP is disabled (weights frozen, only current propagation).

# Returns
- `LayeredNetwork`: The updated network state.
"""
function runlayers!(net::LayeredNetwork, dt::Float64, duration::Float64; t0::Float64=0.0, inputfn=nothing, callback=nothing, freeze_at::Float64=Inf)
    nsteps = Int(round(duration / dt))
    nlayers = length(net.neuronlayers)

    # Pre-allocate fired array and trace snapshots
    fired = [falses(net.neuronlayers[i].N) for i in 1:nlayers]
    old_pretraces = [copy(net.neuronlayers[i].pretraces) for i in 1:nlayers]
    old_posttraces = [copy(net.neuronlayers[i].posttraces) for i in 1:nlayers]

    for step in 1:nsteps
        t = t0 + (step - 1) * dt
        freeze = t >= freeze_at

        # Apply external input to the input layer(s)
        if inputfn !== nothing && nlayers > 0
            input_val = inputfn(t)
            if isa(input_val, Number)
                net.neuronlayers[1].i .+= input_val
            elseif isa(input_val, AbstractVector)
                for (idx, val) in enumerate(input_val)
                    if idx > nlayers break end
                    layer = net.neuronlayers[idx]
                    if (!layer.isreverse && val > 0) || (layer.isreverse && val < 0)
                        layer.i .+= abs(val)
                    end
                end
            else
                error("inputfn(t) must return either a Number or an AbstractVector")
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
    
        for syn in net.synapselayers
            prelayer = net.neuronlayers[syn.pre_idx]
            postlayer = net.neuronlayers[syn.post_idx]
            prefired = fired[syn.pre_idx]

            if any(prefired)
                propagate!(postlayer, syn, prefired, old_posttraces[syn.post_idx]; freeze=freeze)
            end

            if any(fired[syn.post_idx])
                update_post!(syn, fired[syn.post_idx], old_pretraces[syn.pre_idx]; freeze=freeze)
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
    layer.i .*= exp(-dt / layer.τ_s)
    layer.pretraces .*= exp(-dt / layer.τ_pretrace)
    layer.posttraces .*= exp(-dt / layer.τ_posttrace)

    # Refractory handling
    refmask = layer.t_ref .> 0
    layer.v[refmask] .= layer.V_reset
    @. layer.t_ref = max(0.0, layer.t_ref - dt)

    # LIF dynamics
    active = .!refmask
    if any(active)
        dv = @. (-(layer.v[active] - layer.V_rest) + layer.R_m[active] .* layer.i[active]) / layer.τ_m[active] * dt
        @. layer.v[active] += dv
    end

    # Spiking
    fired = layer.v .>= layer.V_thresh

    if any(fired)
        layer.v[fired] .= layer.V_reset 
        layer.t_ref[fired] .= layer.τ_ref
        layer.pretraces[fired] .+= 1.0
        layer.posttraces[fired] .+= 1.0
        layer.t_lastout[fired] .= t
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
 - `freeze::Bool=false`: If true, skip STDP weight update (only propagate current).
"""
function propagate!(post::NeuronLayer, syn::SynapseLayer, fired::BitArray, post_posttraces::Vector{Float64}; freeze=false)
    any(fired) || return

    # Apply weights
    w_impact = syn.isinhibitory ? -syn.ws : syn.ws
    if post.isreverse && !syn.isinhibitory
        w_impact = -w_impact
    end
    post.i .+= sum(w_impact[:, fired], dims=2)[:]

    if freeze
        return
    end

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
 - `freeze::Bool=false`: If true, skip STDP weight update.
"""
function update_post!(syn::SynapseLayer, postfired::BitArray, pre_pretraces::Vector{Float64}; freeze=false)
    any(postfired) || return

    if freeze
        return
    end

    # STDP LTP
    ltp = syn.learningrate .* (postfired * pre_pretraces') .* (1.0 .- syn.ws ./ syn.wmax)
    active = syn.ws .> 0
    syn.ws[active] .= min.(syn.wmax, syn.ws[active] .+ ltp[active])
end

end # module Layers