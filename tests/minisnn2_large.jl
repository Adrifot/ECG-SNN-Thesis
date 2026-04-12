include("../modules/Layers.jl")
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers: update!, propagate!, update_post!

using Plots
using Statistics
plotly()

n_neurons = 50

input_template = Neuron("input";
    R_m=0.5, τ_m=3.0, τ_s=50.0, τ_ref=4.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

output_template = Neuron("output"; 
    R_m=3.0, τ_m=5.0, τ_s=5.0, τ_ref=2.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

input_layer = NeuronLayer(n_neurons, input_template; name="input")
output_layer = NeuronLayer(n_neurons, output_template; name="output")

synapse_template = Synapse(1, 2; learningrate=0.05, w=0.5, wmax=1.0)
synapse_layer = SynapseLayer(input_layer, output_layer, synapse_template;
                             randomweights=true, weightscale=0.9)

constant_input(t; amp=0.002, t0=5.0) = t >= t0 ? amp : 0.0

duration = 50.0
dt = 0.01

nsteps = Int(round(duration / dt))
n_layers = 2

voltage_trace = zeros(n_layers, n_neurons, nsteps)
current_trace = zeros(n_layers, n_neurons, nsteps)
weight_trace = zeros(n_neurons, n_neurons, nsteps)
pre_trace_log = zeros(n_layers, n_neurons, nsteps)
time_axis = zeros(nsteps)

callback = function(t, input_layer, output_layer, syn_layer, step)
    time_axis[step] = t
    voltage_trace[1, :, step] = input_layer.vs
    current_trace[1, :, step] = input_layer.is
    pre_trace_log[1, :, step] = input_layer.pretraces

    voltage_trace[2, :, step] = output_layer.vs
    current_trace[2, :, step] = output_layer.is
    pre_trace_log[2, :, step] = output_layer.pretraces

    weight_trace[:, :, step] = syn_layer.ws
end

function run_layers!(input_layer, output_layer, syn_layer, input_fn, dt, duration; callback=nothing)
    nsteps = Int(round(duration / dt))
    for step in 1:nsteps
        t = (step - 1) * dt

        for i in 1:input_layer.N
            input_layer.is[i] += input_fn(t)
        end

        fired_pre = update!(input_layer, dt, t)
        fired_post = update!(output_layer, dt, t)

        propagate!(output_layer, syn_layer, fired_pre)
        update_post!(input_layer, syn_layer, fired_post)

        if callback !== nothing
            callback(t, input_layer, output_layer, syn_layer, step)
        end
    end
end

run_layers!(input_layer, output_layer, synapse_layer, constant_input, dt, duration; callback=callback)

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

    # Weight panel - mean weight evolution
    pw = plot(xlabel="Time", ylabel="Weight",
              title="STDP Mean Weight Evolution",
              legend=:topright, size=(900, 200))
    w_mean = dropdims(mean(weight_trace, dims=(1,2)), dims=(1,2))
    w_std = dropdims(std(weight_trace, dims=(1,2)), dims=(1,2))
    plot!(time_axis, w_mean, label="Mean weight", ribbon=w_std, linewidth=2)
    hline!([1.0], line=:dash, color=:red, label="w_max")

    # Final weight distribution
    pw_hist = plot(xlabel="Weight", ylabel="Count",
                   title="Final Weight Distribution",
                   legend=false, size=(400, 200))
    histogram!(weight_trace[:, :, end], bins=30, alpha=0.7)

    p = plot(pv, pw, pw_hist, layout=(3, 1), link=:x, size=(900, 700))
    display(p)
    savefig(p, "test2_large_$(replace(input_name, " " => "_")).png")
end

plot_results(time_axis, voltage_trace, weight_trace, "constant input", output_layer, n_neurons)