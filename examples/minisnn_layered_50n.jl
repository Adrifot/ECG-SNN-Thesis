include("../modules/Layers.jl")
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers.Utils
using .Layers: update!, propagate!, update_post!

using Plots
using Statistics
plotly()

n_neurons = 50

input_template = Neuron("input";
    R_m=0.1, τ_m=2.0, τ_s=60.0, τ_ref=4.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

output_template = Neuron("output"; 
    R_m=3.0, τ_m=6.5, τ_s=6.0, τ_ref=2.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

input_layer = NeuronLayer(n_neurons, input_template; name="input")
output_layer = NeuronLayer(n_neurons, output_template; name="output")

synapse_template = Synapse(1, 2; learningrate=0.05, w=0.5, wmax=1.0)
synapse_layer = SynapseLayer(input_layer, output_layer, synapse_template;
                             dist=UniformDist(0.1, 0.9), density=1.0, pre_idx=1, post_idx=2)

constant_input(t; amp=0.005, t0=5.0) = t >= t0 ? amp : 0.0

duration = 100.0
dt = 0.01

nsteps = Int(round(duration / dt))
n_layers = 2

voltage_trace = zeros(n_layers, n_neurons, nsteps)
current_trace = zeros(n_layers, n_neurons, nsteps)
weight_trace = zeros(n_neurons, n_neurons, nsteps)
pre_trace_log = zeros(n_layers, n_neurons, nsteps)
time_axis = zeros(nsteps)

callback = function(t, net, step)
    input_layer = net.neuronlayers[1]
    output_layer = net.neuronlayers[2]

    time_axis[step] = t
    voltage_trace[1, :, step] = input_layer.v
    current_trace[1, :, step] = input_layer.i
    pre_trace_log[1, :, step] = input_layer.pretraces

    voltage_trace[2, :, step] = output_layer.v
    current_trace[2, :, step] = output_layer.i
    pre_trace_log[2, :, step] = output_layer.pretraces

    weight_trace[:, :, step] = net.synapselayers[1].ws
end

net = LayeredNetwork([input_layer, output_layer], [synapse_layer])

runlayers!(net, dt, duration; inputfn=constant_input, callback=callback)

function plot_results(time_axis, voltage_trace, weight_trace, input_name, output_layer, n_neurons)
    # Voltage panel - show mean and std
    pv = plot(xlabel="Time", ylabel="Voltage",
              title="$input_name — Membrane Potentials (mean ± std)",
              legend=:topright, size=(900, 300))
    for layer_idx in 1:2
        v_mean = dropdims(mean(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        v_std = dropdims(std(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        layer_name = layer_idx == 1 ? "input" : "output"
        plot!(time_axis, v_mean, label=layer_name, ribbon=v_std, linewidth=2)
    end
    hline!([output_layer.V_thresh], line=:dash, color=:red, label="Threshold")

    # Weight panel - mean weight 
    pw = plot(xlabel="Time", ylabel="Weight",
              title="STDP Mean Weight Evolution",
              legend=:topright, size=(900, 200))
    w_mean = dropdims(mean(weight_trace, dims=(1,2)), dims=(1,2))
    w_std = dropdims(std(weight_trace, dims=(1,2)), dims=(1,2))
    plot!(time_axis, w_mean, label="Mean weight", ribbon=w_std, linewidth=2)
    hline!([1.0], line=:dash, color=:red, label="w_max")

    # Final weight distribution
    pw_hist = histogram(weight_trace[:, :, end];
                    bins=10, alpha=0.7,
                    xlabel="Weight", ylabel="Count",
                    title="Final Weight Distribution",
                    legend=false,
                    xlim=(0, 1))

    p = plot(pv, pw, pw_hist, layout=(3, 1), size=(900, 700))
    display(p)
    savefig(p, joinpath(@__DIR__, "../docs/imgs/minisnn_layered_50n_$(replace(input_name, " " => "_")).png"))
end

plot_results(time_axis, voltage_trace, weight_trace, "constant input", output_layer, n_neurons)