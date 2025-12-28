using Plots

gr()

mutable struct Neuron
    τ_m::Float64 # Time constant
    τ_ref::Float64 # Refractory period
    V_rest::Float64 # Resting potential
    V_thresh::Float64 # Threshold potential
    V_reset::Float64 # Reset potential
    R_m::Float64 # Resistance
    v::Float64 # Current membrane potential
    t_ref::Float64 # Current remaining refractory time
end

function Neuron(; τ_m=10.0, τ_ref=2.0, V_rest=-75.0, V_thresh=-55.0, 
                V_reset=-75.0, R_m=1.0)
    return Neuron(τ_m, τ_ref, V_rest, V_thresh, V_reset, R_m, V_rest, 0.0)
end

function update!(n::Neuron, I_ext::Float64, dt::Float64)
    # If in refractory period, voltage stuck at V_ref
    if n.t_ref > 0
        n.v = n.V_reset
        n.t_ref -= dt
        return false
    end

    # LIF Equation
    dv = (-(n.v - n.V_rest) + n.R_m * I_ext) / n.τ_m * dt
    n.v += dv

    # Spike check
    if n.v ≥ n.V_thresh
        n.v = n.V_reset
        n.t_ref = n.τ_ref
        return true
    end

    return false
end

n = Neuron()
dt = 0.1
duration = 100.0
times = 0:dt:duration
volts = Float64[]

for t in times
    I_ext = t > 10.0 ? 25.0 : 0.0 # Stimulation starts after 10ms
    fired = update!(n, I_ext, dt)
    push!(volts, n.v)
end

p = plot(times, volts, 
     linecolor=:steelblue, 
     linewidth=2, 
     title="LIF Neuron Stimulation",
     xlabel="Time (ms)", 
     ylabel="Membrane Potential (mV)",
     label="Voltage")

hline!([n.V_thresh], line=:dash, color=:red, label="Threshold")

display(p)
println("Press Enter to close...")
readline()