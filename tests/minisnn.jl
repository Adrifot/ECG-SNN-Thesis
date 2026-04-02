include("../modules/Networks.jl")
# WORK IN PROGRESS
using .Networks
using .Networks.Neurons
using .Networks.Synapses

using Plots
plotly()

net = Network()

addneuron!(net, Neuron("input"; R_m=1.0, τ_m=5.0, τ_s=5.0, τ_ref=4.0))
addneuron!(net, Neuron("output"; R_m=3.0, τ_m=5.0, τ_s=5.0, τ_ref=2.0))
addsynapse!(net, "input", "output"; learningrate=0.05, τ_pre=30.0, τ_post=10.0)

constant_input(t; amp=0.5, t0=5.0) = t >= t0 ? amp : 0.0
linear_input(t; slope=0.1, t0=0.0) = t >= t0 ? slope*(t-t0) : 0.0
square_input(t; amp=5.0, period=40.0, dutycycle=0.5) = (t%period)/period < dutycycle ? amp : 0.0

duration = 500.0
dt = 0.01

nsteps = Int(round(duration / dt))
n_neurons = length(net.neurons)
n_synapses = length(net.synapses)

voltage_trace = zeros(n_neurons, nsteps)
current_trace = zeros(n_neurons, nsteps)
weight_trace = zeros(n_synapses, nsteps)
time_axis = zeros(nsteps)

callback = function(t, net, step)
    time_axis[step] = t
    for i in 1:n_neurons
        voltage_trace[i, step] = net.neurons[i].v
        current_trace[i, step] = net.neurons[i].i
    end
    for s in 1:n_synapses
        weight_trace[s, step] = net.synapses[s].w
    end
end

run!(net, constant_input, [1], dt, duration; callback=callback)

# Plotting
function plot_results(time_axis, voltage_trace, weight_trace, spikes, net, input_name)
    n_neurons = length(net.neurons)


    pv = plot(xlabel="Time", ylabel="Voltage",
              title="$input_name — Membrane Potentials",
              legend=:topright, size=(900, 300))
    for i in 1:n_neurons
        plot!(time_axis, voltage_trace[i, :], label=net.neurons[i].name, linewidth=2)
    end
    hline!([net.neurons[1].V_thresh], line=:dash, color=:red, label="Threshold")

    # Weight panel
    pw = plot(time_axis, weight_trace[1, :], label="Synapse 1",
              xlabel="Time", ylabel="Weight",
              title="STDP Weight Evolution",
              legend=:topright, size=(900, 200))
    for s in 2:size(weight_trace, 1)
        plot!(time_axis, weight_trace[s, :], label="Synapse $s")
    end
    hline!([net.synapses[1].wmax], line=:dash, color=:red, label="w_max")

    # Raster panel
    pr = plot(xlabel="Time", ylabel="Neuron", title="Spike Raster", legend=:topright, size=(900, 200))
    for (i, neuron) in enumerate(net.neurons)
        neuron_spikes = [s.time for s in spikes if s.src_name == neuron.name]
        if !isempty(neuron_spikes)
            scatter!(neuron_spikes, fill(i, length(neuron_spikes)), markersize=3, color=:black, label=neuron.name)
        end
    end
    yticks_vals = 1:n_neurons
    yticks_labels = [n.name for n in net.neurons]
    plot!(yticks=(yticks_vals, yticks_labels))

    p = plot(pv, pw, pr, layout=(3, 1), link=:x, size=(900, 700), legend=:topright)
    display(p)
    savefig(p, "test_$(input_name).png")
end

plot_results(time_axis, voltage_trace, weight_trace, net.spikelog, net, "constant input")