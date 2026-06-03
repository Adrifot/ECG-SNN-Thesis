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

PATIENT = "001"
SESSION = "s0014lre"

Δ = 0.1
fs = 1000.0
N = 50

up_template = Neuron("up_input"; R_m=0.5, τ_m=2.0, τ_s=60.0, τ_ref=4.0,
                τ_pretrace=20.0, τ_posttrace=10.0)

down_template = Neuron("down_input"; R_m=0.5, τ_m=2.0, τ_s=60.0, τ_ref=4.0,
                τ_pretrace=20.0, τ_posttrace=10.0)

out_template = Neuron("output"; R_m=3.0, τ_m=6.0, τ_s=6.0, τ_ref=2.0,
                τ_pretrace=20.0, τ_posttrace=10.0)

exc_syn_template = Synapse(1, 2; learningrate=0.05, wmax=1.0)

inhib_template = Synapse(1, 1; learningrate=0.0, wmax=1.0, isinhibitory=true)

up_layer = NeuronLayer(N, up_template;   name="up_input",   V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)
down_layer = NeuronLayer(N, down_template; name="down_input", V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)
out_layer = NeuronLayer(N, out_template;  name="output",     V_thresh_dev=0.1, R_m_dev=0.1, τ_m_dev=0.15)

up_to_out = SynapseLayer(up_layer, out_layer, exc_syn_template;
    dist=NormalDist(0.5, 0.2), density=0.75, pre_idx=1, post_idx=3)

down_to_out = SynapseLayer(down_layer, out_layer, exc_syn_template;
    dist=NormalDist(0.5, 0.2), density=0.75, pre_idx=2, post_idx=3)

inhib_up = SynapseLayer(up_layer,   up_layer,   inhib_template;
    dist=UniformDist(0.25, 0.5), density=0.4, pre_idx=1, post_idx=1)
inhib_down = SynapseLayer(down_layer, down_layer, inhib_template;
    dist=UniformDist(0.25, 0.5), density=0.4, pre_idx=2, post_idx=2)
inhib_out = SynapseLayer(out_layer,  out_layer,  inhib_template;
    dist=UniformDist(0.25, 0.5), density=0.4, pre_idx=3, post_idx=3)

net = LayeredNetwork(
    [up_layer, down_layer, out_layer],
    [up_to_out, down_to_out, inhib_up, inhib_down, inhib_out]
)

spiketrain, siglen, filtsig = get_spiketrain(PATIENT, SESSION; Δ=Δ)
tsim = min(25.0, siglen / fs)
dt = 0.001
nsteps = Int(round(tsim / dt))

pulse_amp = 25.0
up_pulses   = zeros(nsteps)
down_pulses = zeros(nsteps)

for spike in spiketrain
    step = Int(floor(spike.time)) + 1
    if 1 <= step <= nsteps
        if spike.polarity
            up_pulses[step]   += pulse_amp   
        else
            down_pulses[step] += pulse_amp   
        end
    end
end

input_fn = t -> begin
    idx = min(max(Int(floor(t / dt)) + 1, 1), nsteps)
    return [up_pulses[idx], down_pulses[idx]]
end


L = 3  
voltage_trace   = zeros(L, N, nsteps)
current_trace   = zeros(L, N, nsteps)
weight_trace_up   = zeros(N, N, nsteps)
weight_trace_down = zeros(N, N, nsteps)
time_axis = zeros(nsteps)

callback = function(t, net, step)
    up   = net.neuronlayers[1]
    down = net.neuronlayers[2]
    out  = net.neuronlayers[3]

    time_axis[step] = t
    voltage_trace[1, :, step] = up.v
    voltage_trace[2, :, step] = down.v
    voltage_trace[3, :, step] = out.v

    current_trace[1, :, step] = up.i
    current_trace[2, :, step] = down.i
    current_trace[3, :, step] = out.i

    weight_trace_up[:, :, step]   = net.synapselayers[1].ws  # up->out
    weight_trace_down[:, :, step] = net.synapselayers[2].ws  # down->out
end

#
# RUN
#

println("=== Before runlayers! ===")
println("Up->Out weights — min: $(round(minimum(net.synapselayers[1].ws), digits=3)), " *
        "max: $(round(maximum(net.synapselayers[1].ws), digits=3)), " *
        "mean: $(round(mean(net.synapselayers[1].ws), digits=3))")
println("Down->Out weights — min: $(round(minimum(net.synapselayers[2].ws), digits=3)), " *
        "max: $(round(maximum(net.synapselayers[2].ws), digits=3)), " *
        "mean: $(round(mean(net.synapselayers[2].ws), digits=3))")

p1 = heatmap(net.synapselayers[1].ws, clims=(0.0, 1.0), title="up→out weights (final)")
p2 = heatmap(net.synapselayers[2].ws, clims=(0.0, 1.0), title="down→out weights (final)")
display(plot(p1, p2, layout=(1, 2), size=(1200, 500)))

runlayers!(net, dt, tsim; inputfn=input_fn, callback=callback)

println("\n=== After runlayers! ===")
println("Up->Out weights — min: $(round(minimum(net.synapselayers[1].ws), digits=3)), " *
        "max: $(round(maximum(net.synapselayers[1].ws), digits=3)), " *
        "mean: $(round(mean(net.synapselayers[1].ws), digits=3))")
println("Down->Out weights — min: $(round(minimum(net.synapselayers[2].ws), digits=3)), " *
        "max: $(round(maximum(net.synapselayers[2].ws), digits=3)), " *
        "mean: $(round(mean(net.synapselayers[2].ws), digits=3))")

#
# PLOTTING
#

function plot_dual_results(time_axis, voltage_trace, w_up, w_down, out_layer, n_neurons)
    # Voltage panel 
    pv = plot(xlabel="Time (s)", ylabel="Voltage",
              title="Dual Input Pathway — Membrane Potentials (mean ± std)",
              legend=:topright, size=(900, 350))
    labels = ["up input", "down input", "output"]
    colors = [:green, :red, :blue]
    for layer_idx in 1:3
        v_mean = dropdims(mean(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        v_std  = dropdims(std(voltage_trace[layer_idx, :, :], dims=1), dims=1)
        plot!(time_axis, v_mean, label=labels[layer_idx], color=colors[layer_idx],
              ribbon=v_std, linewidth=2)
    end
    hline!([out_layer.V_thresh], line=:dash, color=:black, label="Threshold")

    # Weight panel
    pw = plot(xlabel="Time (s)", ylabel="Weight",
              title="STDP Mean Weight Evolution (up vs down pathway)",
              legend=:topright, size=(900, 200))
    for (name, w_trace, col) in [("up→out", w_up, :green), ("down→out", w_down, :red)]
        w_mean = dropdims(mean(w_trace, dims=(1,2)), dims=(1,2))
        w_std  = dropdims(std(w_trace, dims=(1,2)), dims=(1,2))
        plot!(time_axis, w_mean, label=name, color=col, ribbon=w_std, linewidth=2)
    end
    hline!([1.0], line=:dash, color=:black, label="w_max")

    # Weight histograms
    pw_hist = histogram(w_up[:, :, end]; bins=10, alpha=0.5,
                        color=:green, label="up→out", normalize=true)
    histogram!(pw_hist, w_down[:, :, end]; bins=10, alpha=0.5,
               color=:red, label="down→out", normalize=true,
               xlabel="Weight", ylabel="Density",
               title="Final Weight Distribution (both pathways)",
               xlim=(0, 1))

    p = plot(pv, pw, pw_hist, layout=(3, 1), size=(900, 750))
    display(p)
    savefig(p, joinpath(@__DIR__, "../docs/imgs/2L_ECG_inhib_dual.png"))
end

plot_dual_results(time_axis, voltage_trace, weight_trace_up, weight_trace_down, out_layer, N)

# Heatmaps
p1 = heatmap(net.synapselayers[1].ws, clims=(0.0, 1.0), title="up→out weights (final)")
p2 = heatmap(net.synapselayers[2].ws, clims=(0.0, 1.0), title="down→out weights (final)")
display(plot(p1, p2, layout=(1, 2), size=(1200, 500)))
