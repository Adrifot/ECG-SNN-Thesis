"""
Neuron.jl

Definitions for `Spike`, `Neuron` and their functions.
"""

module Neuron

export Neuron, update!, Spike, OutputSpike

"""
A spike emitted by a neuron at a given time with a polarity.

# Fields
- `time`: The time (ms) at which the spike occurs.
- `polarity`: Spike polarity; `true` for up, `false` for down.
"""
struct Spike 
    time::Float64
    polarity::Bool
end

"""
A spike recorded from a neuron, annotated with its source name.

# Fields
- `time`: The time (ms) at which the spike occurred.
- `neuron_name`: Name/identifier of the neuron that fired.
"""
struct OutputSpike
    time::Float64
    neuron_name::String
end

"""
A leaky integrate-and-fire (LIF) neuron with optional polarity sensitivity.

# Fields
- `name::String`: Identifier of the neuron.
- `τ_m::Float64`: Membrane time constant (ms).
- `τ_ref::Float64`: Refractory period (ms).
- `V_rest::Float64`: Resting membrane potential (mV).
- `V_thresh::Float64`: Spike threshold potential (mV).
- `V_reset::Float64`: Reset potential after spike (mV).
- `R_m::Float64`: Membrane resistance.
- `v::Float64`: Current membrane potential (mV).
- `t_ref::Float64`: Remaining refractory time (ms).
- `i_ext::Float64`: Current synaptic input.
- `τ_s::Float64`: Synaptic decay time constant (ms).
- `w::Float64`: Synaptic weight.
- `is_reverse::Bool`: Whether this neuron responds to reverse polarity spikes.
"""
mutable struct Neuron
    name::String
    τ_m::Float64
    τ_ref::Float64
    V_rest::Float64
    V_thresh::Float64
    V_reset::Float64
    R_m::Float64
    v::Float64
    t_ref::Float64
    i_ext::Float64
    τ_s::Float64
    w::Float64
    is_reverse::Bool
end

"""
Construct a new `Neuron` instance with default LIF parameters.  

# Keyword Arguments
- `name`: Neuron identifier.
- `τ_m`: Membrane time constant.
- `τ_ref`: Refractory period.
- `V_rest`: Resting potential.
- `V_thresh`: Threshold potential.
- `V_reset`: Reset potential.
- `R_m`: Membrane resistance.
- `τ_s`: Synaptic decay constant.
- `w`: Synaptic weight.
- `is_reverse`: Sensitivity to reverse-polarity spikes.

# Returns
- A new `Neuron` instance.
"""
function Neuron(; name="neuron", τ_m=20.0, τ_ref=2.0, V_rest=-70.0, 
                V_thresh=-50.0, V_reset=-70.0, R_m=1.0, 
                τ_s=5.0, w=15.0, is_reverse=false)
    return Neuron(name, τ_m, τ_ref, V_rest, V_thresh, V_reset, 
                  R_m, 0.0, 0.0, 0.0, τ_s, w, is_reverse)
end

"""
    update!(n::Neuron, spike_type::Int, dt::Float64) -> Bool

Advance the state of a neuron by one time step `dt` with optional synaptic input.

# Arguments
- `n`: The neuron to update.
- `spike_type`: Incoming spike type. Use `0` for no spike, positive for excitatory, negative for inhibitory.
- `dt`: Time step (ms).

# Returns
- `true` if the neuron fired a spike in this time step.
- `false` otherwise.

# Behavior
1. Decays the synaptic current (`i_ext`) according to `τ_s`.
2. Injects new synaptic current if `spike_type` polarity matches `is_reverse`.
3. Handles refractory period: keeps membrane potential at `V_reset` if `t_ref > 0`.
4. Updates membrane potential using the LIF differential equation.
5. Checks threshold: if `v ≥ V_thresh`, neuron spikes, resets `v` to `V_reset`, sets refractory time, and clears `i_ext`.
"""
function update!(n::Neuron, spike_type::Int, dt::Float64)
    n.i_ext += (-n.i_ext / n.τ_s) * dt

    if spike_type != 0
        mult = n.is_reverse ? -1 : 1
        if (spike_type * mult) > 0
            n.i_ext += n.w 
        end
    end

    if n.t_ref > 0
        n.v = n.V_reset
        n.t_ref -= dt
        return false
    end

    dv = (-(n.v - n.V_rest) + n.R_m * n.i_ext) / n.τ_m * dt
    n.v += dv

    if n.v ≥ n.V_thresh
        n.v = n.V_reset
        n.t_ref = n.τ_ref
        n.i_ext = 0 
        return true
    end

    return false
end

end # module Neuron