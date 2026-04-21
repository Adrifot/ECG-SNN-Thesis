include("../modules/Signals.jl")

using .Signals
using Plots

gr()


PATIENT = "121"
SESSION = "s0311lre"

function plot_spiketrain_demo(signal, spiketrain; fs=1000, duration=2.0)
    max_idx = min(Int(duration * fs), length(signal))
    time_axis = (0:max_idx-1) ./ fs
    short_signal = signal[1:max_idx]
    
    up_times = [s.time / fs for s in spiketrain if s.polarity && s.time <= max_idx]
    down_times = [s.time / fs for s in spiketrain if !s.polarity && s.time <= max_idx]

    p1 = plot(time_axis, short_signal, 
              ylabel="mV", title="Lead II (Filtered)", 
              lc=:blue, legend=false)

    p2 = scatter(up_times, fill(1, length(up_times)), 
                 markershape=:vline, markerstrokewidth=2, markersize=8,
                 color=:green, label="Up Spike", alpha=0.8, legend=false)
    
    scatter!(p2, down_times, fill(0, length(down_times)), 
                 markershape=:vline, markerstrokewidth=2, markersize=8,
                 color=:red, label="Down Spike", alpha=0.8, legend=false)

    plot!(p2, yticks=([0, 1], ["Down", "Up"]), ylims=(-0.5, 1.5),
          xlabel="Time (seconds)", ylabel="Direction", legend=false)

    return plot(p1, p2, layout=(2, 1), size=(900, 500), link=:x)
end

raw_signal = load_raw_signal(PATIENT, SESSION)
filtered_signal = get_filtered_signal(raw_signal)
spiketrain = delta_modulation(filtered_signal; Δ=100.0)

p = plot_spiketrain_demo(filtered_signal, spiketrain)
display(p)
println("Press Enter to close plot...")
readline()
savefig(p, joinpath(@__DIR__, "../docs/imgs/ECG_to_spiketrain.png"))
println("Plot saved to /docs/imgs/")