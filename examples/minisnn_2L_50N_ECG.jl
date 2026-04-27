include("../modules/Layers.jl")
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers.Utils
using .Layers: update!, propagate!, update_post!

include("../modules/Signals.jl")

using .Signals

using Plots
using Statistics
plotly()

PATIENT = "121"
SESSION = "s0311lre"
Δ = 100.0
fs = 1000.0

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
synapse_layer = SynapseLayer(input_layer, output_layer, synapse_template; dist=NormalDist(0.5, 0.1), density=0.5)

spiketrain, signal_length, filtered_signal = get_spiketrain(PATIENT, SESSION; Δ=Δ)

duration = min(25.0, signal_length / fs)
dt = 0.001

nsteps = Int(round(duration / dt))
n_layers = 2

pulse_amp = 10.0
input_pulses = zeros(nsteps)
for spike in spiketrain
    step = Int(floor(((spike.time - 1.0) / fs) / dt)) + 1
    if 1 <= step <= nsteps
        input_pulses[step] += spike.polarity ? pulse_amp : pulse_amp # HACK: Consider polarity in the future
    end
end

input_fn = t -> begin
    idx = min(max(Int(floor(t / dt)) + 1, 1), nsteps)
    return input_pulses[idx]
end

voltage_trace = zeros(n_layers, n_neurons, nsteps)
current_trace = zeros(n_layers, n_neurons, nsteps)
weight_trace = zeros(n_neurons, n_neurons, nsteps)
pre_trace_log = zeros(n_layers, n_neurons, nsteps)
time_axis = zeros(nsteps)

callback = function(t, net, step)
    input_layer = net.neuronlayers[1]
    output_layer = net.neuronlayers[2]

    time_axis[step] = t
    voltage_trace[1, :, step] = input_layer.vs
    current_trace[1, :, step] = input_layer.is
    pre_trace_log[1, :, step] = input_layer.pretraces

    voltage_trace[2, :, step] = output_layer.vs
    current_trace[2, :, step] = output_layer.is
    pre_trace_log[2, :, step] = output_layer.pretraces

    weight_trace[:, :, step] = net.synapselayers[1].ws
end

net = LayeredNetwork([input_layer, output_layer], [synapse_layer])

runlayers!(net, dt, duration; inputfn=(t, layer_idx) -> (layer_idx == 1 ? input_fn(t) : 0.0), callback=callback)

# PLOTTING -------------------------------------------------------------------------------------

function plot_results(time_axis, voltage_trace, weight_trace, input_name, output_layer, n_neurons)
    pv = plot(xlabel="Time (s)", ylabel="Voltage",
              title="$input_name — Membrane Potentials (mean ± std)",
              legend=:topright, size=(900, 300))
    for layer_idx in 1:2
        v_mean = dropdims(mean(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        v_std = dropdims(std(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        layer_name = layer_idx == 1 ? "input" : "output"
        plot!(time_axis, v_mean, label=layer_name, ribbon=v_std, linewidth=2)
    end
    hline!([output_layer.V_thresh], line=:dash, color=:red, label="Threshold")

    pw = plot(xlabel="Time (s)", ylabel="Weight",
              title="STDP Mean Weight Evolution",
              legend=:topright, size=(900, 200))
    w_mean = dropdims(mean(weight_trace, dims=(1,2)), dims=(1,2))
    w_std = dropdims(std(weight_trace, dims=(1,2)), dims=(1,2))
    plot!(time_axis, w_mean, label="Mean weight", ribbon=w_std, linewidth=2)
    hline!([1.0], line=:dash, color=:red, label="w_max")

    pw_hist = histogram(weight_trace[:, :, end];
                    bins=10, alpha=0.7,
                    xlabel="Weight", ylabel="Count",
                    title="Final Weight Distribution",
                    legend=false,
                    xlim=(0, 1))

    p = plot(pv, pw, pw_hist, layout=(3, 1), size=(900, 700))
    display(p)
    savefig(p, joinpath(@__DIR__, "../docs/imgs/minisnn_layered_50n_ECG.png"))
end

plot_results(time_axis, voltage_trace, weight_trace, "ECG delta-modulated spiketrain", output_layer, n_neurons)

pp = plot(heatmap(net.synapselayers[1].ws))
display(pp)
