include("../modules/Networks.jl")
using .Networks
using .Networks.Neurons
using .Networks.Synapses

using Plots
plotly()

net = Network()

addneuron!(net, Neuron("input";
    R_m=0.1, τ_m=2.0, τ_s=60.0, τ_ref=4.0, 
    τ_pretrace=30.0, τ_posttrace=10.0))

addneuron!(net, Neuron("output"; 
    R_m=3.0, τ_m=6.5, τ_s=6.0, τ_ref=2.0, 
    τ_pretrace=30.0, τ_posttrace=10.0))

addsynapse!(net, "input", "output"; learningrate=0.05, w=0.5, wmax=1.0)

# Input function
constant_input(t; amp=0.005, t0=5.0) = t >= t0 ? amp : 0.0

duration = 100.0
dt = 0.01

nsteps = Int(round(duration / dt))
n_neurons = length(net.neurons)
n_synapses = length(net.synapses)

# Data containers
voltage_trace = zeros(n_neurons, nsteps)
current_trace = zeros(n_neurons, nsteps)
weight_trace = zeros(n_synapses, nsteps)
pre_trace_log = zeros(n_neurons, nsteps) 
time_axis = zeros(nsteps)

callback = function(t, net, step)
    time_axis[step] = t
    for i in 1:n_neurons
        voltage_trace[i, step] = net.neurons[i].v
        current_trace[i, step] = net.neurons[i].i
        pre_trace_log[i, step] = net.neurons[i].pretrace
    end
    for s in 1:n_synapses
        weight_trace[s, step] = net.synapses[s].w
    end
end

run!(net, constant_input, [1], dt, duration; callback=callback)

# Plotting 
function plot_results(time_axis, voltage_trace, weight_trace, spikes, net, input_name)
    n_neurons = length(net.neurons)

    # Voltage panel
    pv = plot(xlabel="Time", ylabel="Voltage",
              title="$input_name — Membrane Potentials",
              legend=:topright, size=(900, 300))
    for i in 1:n_neurons
        plot!(time_axis, voltage_trace[i, :], label=net.neurons[i].name, linewidth=2)
    end

    hline!([net.neurons[1].V_thresh], line=:dash, color=:red, label="Threshold")

    # Weight panel
    pw = plot(xlabel="Time", ylabel="Weight",
              title="STDP Weight Evolution",
              legend=:topright, size=(900, 200))
    for s in 1:size(weight_trace, 1)
        plot!(time_axis, weight_trace[s, :], label="Synapse $s", linewidth=2)
    end
    
    if !isempty(net.synapses)
        hline!([net.synapses[1].wmax], line=:dash, color=:red, label="w_max")
    end

    # Raster panel
    pr = plot(xlabel="Time", ylabel="Neuron", title="Spike Raster", legend=:topright, size=(900, 200))
    for (i, neuron) in enumerate(net.neurons)
        neuron_spikes = [s.time for s in spikes if s.src_name == neuron.name]
        if !isempty(neuron_spikes)
            scatter!(neuron_spikes, fill(i, length(neuron_spikes)), 
                     markersize=4, color=:black, label=neuron.name, markerstrokewidth=0)
        end
    end
    yticks_vals = 1:n_neurons
    yticks_labels = [n.name for n in net.neurons]
    plot!(yticks=(yticks_vals, yticks_labels), ylims=(0.5, n_neurons + 0.5))

    p = plot(pv, pw, pr, layout=(3, 1), link=:x, size=(900, 800))
    display(p)
    savefig(p, joinpath(@__DIR__, "../docs/imgs/minisnn_$(replace(input_name, " " => "_")).png"))
    println("Plot saved to /docs/imgs/")
end

plot_results(time_axis, voltage_trace, weight_trace, net.spikelog, net, "constant input")