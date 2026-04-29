include("../modules/Layers.jl")
include("../modules/Signals.jl")

using .Signals
using .Layers
using .Layers.Neurons
using .Layers.Synapses
using .Layers.Utils
using .Layers: update!, propagate!, update_post!

using Plots
using Statistics

plotly()

PATIENT = "121"
SESSION = "s0311lre"

Δ = 100.0
fs = 1000.0
N = 50

in_template = Neuron("input"; R_m=0.5, τ_m=2.0, τ_s=60.0, τ_ref=4.0,
                τ_pretrace=20.0, τ_posttrace=10.0)

out_template = Neuron("output"; R_m=3.0, τ_m=6.0, τ_s=6.0, τ_ref=2.0,
                τ_pretrace=20.0, τ_posttrace=10.0)

syn_template = Synapse(1, 2; learningrate=0.05, wmax=1.0)

inhib_template = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

inlayer = NeuronLayer(N, in_template; name="input", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
outlayer = NeuronLayer(N, out_template; name="output", V_thresh_dev=0.05, R_m_dev=0.1, τ_m_dev=0.15)
synlayer = SynapseLayer(inlayer, outlayer, syn_template; dist=NormalDist(0.5, 0.2), density=0.75, pre_idx=1, post_idx=2)
inhiblayer1 = SynapseLayer(inlayer, inlayer, inhib_template; dist=ConstantDist(0.15), density=0.75, pre_idx=1, post_idx=1)
inhiblayer2 = SynapseLayer(outlayer, outlayer, inhib_template; dist=ConstantDist(0.15), density=0.75, pre_idx=2, post_idx=2)

net = LayeredNetwork([inlayer, outlayer], [synlayer, inhiblayer1, inhiblayer2])

spiketrain, siglen, filtsig = get_spiketrain(PATIENT, SESSION; Δ=Δ)
tsim = min(25.0, siglen/fs)
dt = 0.001
nsteps = Int(round(tsim/dt))
L = 2

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

voltage_trace = zeros(L, N, nsteps)
current_trace = zeros(L, N, nsteps)
weight_trace = zeros(N, N, nsteps)
pre_trace_log = zeros(L, N, nsteps)
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

println("=== Before runlayers! ===")
println("Weight min: ", minimum(net.synapselayers[1].ws))
println("Weight max: ", maximum(net.synapselayers[1].ws))
println("Weight mean: ", mean(net.synapselayers[1].ws))
println("Weight std: ", std(net.synapselayers[1].ws))

pp = plot(heatmap(net.synapselayers[1].ws, clims=(0.0, 1.0)))
display(pp)

runlayers!(net, dt, tsim; inputfn=input_fn, callback=callback)

println("\n=== After runlayers! ===")
println("Weight min: ", minimum(net.synapselayers[1].ws))
println("Weight max: ", maximum(net.synapselayers[1].ws))
println("Weight mean: ", mean(net.synapselayers[1].ws))
println("Weight std: ", std(net.synapselayers[1].ws))

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
    savefig(p, joinpath(@__DIR__, "../docs/imgs/2L_50N_ECG_inhib.png"))
end

plot_results(time_axis, voltage_trace, weight_trace, "ECG", outlayer, N)

pp = plot(heatmap(net.synapselayers[1].ws, clims=(0.0, 1.0)))
display(pp)
