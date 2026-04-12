include("../modules/Layers.jl")
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers: update!, propagate!, update_post!

using Plots
plotly()

input_template = Neuron("input";
    R_m=0.5, τ_m=3.0, τ_s=50.0, τ_ref=4.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

output_template = Neuron("output"; 
    R_m=3.0, τ_m=5.0, τ_s=5.0, τ_ref=2.0, 
    τ_pretrace=30.0, τ_posttrace=10.0)

input_layer = NeuronLayer(1, input_template; name="input")
output_layer = NeuronLayer(1, output_template; name="output")

synapse_template = Synapse(1, 2; learningrate=0.05, w=0.5, wmax=1.0)
synapse_layer = SynapseLayer(input_layer, output_layer, synapse_template;
                             randomweights=false, weightscale=1.0)

constant_input(t; amp=0.005, t0=5.0) = t >= t0 ? amp : 0.0

duration = 100.0
dt = 0.01

nsteps = Int(round(duration / dt))
n_layers = 2

voltage_trace = zeros(n_layers, nsteps)
current_trace = zeros(n_layers, nsteps)
weight_trace = zeros(nsteps)
pre_trace_log = zeros(n_layers, nsteps)
time_axis = zeros(nsteps)

callback = function(t, input_layer, output_layer, syn_layer, step)
    time_axis[step] = t
    voltage_trace[1, step] = input_layer.vs[1]
    current_trace[1, step] = input_layer.is[1]
    pre_trace_log[1, step] = input_layer.pretraces[1]

    voltage_trace[2, step] = output_layer.vs[1]
    current_trace[2, step] = output_layer.is[1]
    pre_trace_log[2, step] = output_layer.pretraces[1]

    weight_trace[step] = syn_layer.ws[1, 1]
end

function run_layers!(input_layer, output_layer, syn_layer, input_fn, dt, duration; callback=nothing)
    nsteps = Int(round(duration / dt))
    for step in 1:nsteps
        t = (step - 1) * dt

        input_layer.is[1] += input_fn(t)

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

function plot_results(time_axis, voltage_trace, weight_trace, input_name, output_layer)
    pv = plot(xlabel="Time", ylabel="Voltage",
              title="$input_name — Membrane Potentials",
              legend=:topright, size=(900, 300))
    plot!(time_axis, voltage_trace[1, :], label="input", linewidth=2)
    plot!(time_axis, voltage_trace[2, :], label="output", linewidth=2)
    hline!([output_layer.V_thresh], line=:dash, color=:red, label="Threshold")

    pw = plot(xlabel="Time", ylabel="Weight",
              title="STDP Weight Evolution",
              legend=:topright, size=(900, 200))
    plot!(time_axis, weight_trace, label="Synapse", linewidth=2)
    hline!([1.0], line=:dash, color=:red, label="w_max")

    p = plot(pv, pw, layout=(2, 1), link=:x, size=(900, 600))
    display(p)
    savefig(p, "test2_$(replace(input_name, " " => "_")).png")
end

plot_results(time_axis, voltage_trace, weight_trace, "constant input", output_layer)