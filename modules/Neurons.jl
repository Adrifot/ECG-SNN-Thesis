"""
    Neurons.jl

Simulation primitives for LIF neurons and spike representations.

# Provides
- `Spike`: Input/output events received or emitted by the system.
- `Neuron`: The core LIF model with synaptic dynamics.
- `update!`: The state integration function.
- `receive_spike!`: Spike receiving function.
"""
module Neurons

export Neuron, update!, Spike, receive_spike!

"""
    Spike(time, polarity, src_name)

A spike received or emitted by a neuron at a given time with a polarity, annotated with its source.

# Fields
- `time::Float64`: Time at which the spike was received.
- `polarity::Bool`: Polarity of received spike. `true` for upward, `false` for downward (antispike).
- `src_name::String`: Identifier of the source. 
"""
struct Spike 
    time::Float64
    polarity::Bool
    src_name::String
end


"""
    Neuron

A leaky integrate-and-fire (LIF) neuron with optional polarity sensitivity.
All parameters are unitless unless the user enforces a consistent unit system.

# Fields
- `name::String`: Neuron's identifier.
- `V_rest::Float64`: Resting membrane potential.
- `V_thresh::Float64`: Spike threshold potential.
- `V_reset::Float64`: Reset potential.
- `R_m::Float64`: Membrane resistance.
- `τ_m::Float64`: Membrane time constant.
- `τ_s::Float64`: Synaptic decay constant.
- `τ_ref::Float64`: Refractory period.
- `isreverse::Bool`: Whether this neuron responds to antispikes.
- `i::Float64`: Current synaptic input.
- `v::Float64`: Current membrane potential.
- `t_ref::Float64`: Remaining refractory time.
"""
mutable struct Neuron
    name::String
    V_rest::Float64 
    V_thresh::Float64
    V_reset::Float64
    R_m::Float64
    τ_m::Float64
    τ_s::Float64
    τ_ref::Float64
    isreverse::Bool
    i::Float64
    v::Float64
    t_ref::Float64

    @doc"""
        Neuron(name; kwargs...) -> Neuron

    Create a new `Neuron` instance. Inner constructor for the `Neuron` struct.

    # Arguments
    - `name::String`: Neuron's identifier
    - `V_rest::Float64=0.0`: Resting membrane potential.
    - `V_thresh::Float64=1.0`: Spike threshold potential.
    - `V_reset::Float64=0.0`: Reset potential.
    - `R_m::Float64=1.0`: Membrane resistance.
    - `τ_m::Float64=20.0`: Membrane time constant.
    - `τ_s::Float64=5.0`: Synaptic decay constant.
    - `τ_ref::Float64=2.0`: Refractory period.
    - `isreverse::Bool=false`: Whether this neuron responds to antispikes.
    """
    function Neuron(
        name::String; 
        V_rest::Float64 = 0.0,
        V_thresh::Float64 = 1.0,
        V_reset::Float64 = 0.0,
        R_m::Float64 = 1.0,
        τ_m::Float64 = 20.0,
        τ_s::Float64 = 5.0,
        τ_ref::Float64 = 2.0,
        isreverse::Bool = false
    )
        τ_m > 0 || throw(ArgumentError("τ_m must be positive (got $τ_m)"))
        τ_s > 0 || throw(ArgumentError("τ_s must be positive (got $τ_s)"))
        V_thresh > V_rest || throw(ArgumentError("V_thresh must be greater than V_rest"))
        τ_ref >= 0 || throw(ArgumentError("τ_ref cannot be negative (got $τ_ref)"))
        
        return new(name, V_rest, V_thresh, V_reset, R_m, τ_m, τ_s, τ_ref, isreverse, 0.0, V_rest, 0.0)
    end
end

"""
    receive_spike!(n, weight)

Inject current into the neuron. This is where the Synapse's weight is applied.

# Arguments:
- `n::Neuron`: Neuron receiving the spike.
- `weight::Float64`: Applied synaptic weight.
"""
receive_spike!(n::Neuron, weight::Float64) = n.i += weight


"""
    update!(n, dt, t) -> Bool

Advance the state of a `Neuron` by one time step using the LIF model.

# Arguments
- `n::Neuron`: The neuron to update.
- `dt::Float64`: Time step used.
- `t::Float64`: Current simulation time.

# Returns
- `Bool`: `true` if the neuron fires a spike at time `t`, `false` otherwise.

# Behavior
1. **Synaptic current decay**: `n.i` decays exponentially with time constant `n.τ_s`.

2. **Refractory handling**: If the neuron is in its refractory period (`n.t_ref > 0`),
    membrane potential `n.v` is clamped to `V_reset` and no spikes can be generated.

3. **Membrane update**: Membrane potential `n.v` is updated using LIF dynamics.

4. **Spike generation**: If membrane potential `n.v` reaches or exceeds threshold `n.V_thresh`,
    the neuron emits a spike, potential and current are reset, time variables are updated accordingly.
"""
function update!(n::Neuron, dt::Float64, t::Float64)

    # 1. Synaptic current decay
    # i(t) = i(0) * exp(-dt/τ_s)
    n.i *= exp(-dt / n.τ_s)

    # 2. Refractory handling
    if n.t_ref > 0
        n.v = n.V_reset
        n.t_ref -= dt
        return false
    end

    # 3. LIF logic 
    dv = (-(n.v - n.V_rest) + n.R_m * n.i) / n.τ_m * dt
    n.v += dv

    # 4. Spike generation
    if n.v ≥ n.V_thresh
        n.v = n.V_reset
        n.t_ref = n.τ_ref
        n.t_lastout = t 

        return true
    end

    return false
end

end # module Neurons

