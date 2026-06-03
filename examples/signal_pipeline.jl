include("../modules/Signals.jl")
using .Signals
using Statistics
using Plots
plotly()

PATIENT = "105"
SESSION = "s0303lre"
Δ = 0.1
fs = 1000.0

raw = load_raw_signal(PATIENT, SESSION)
filt = get_filtered_signal(raw)
peaks = get_R_peaks(filt; fs=fs)
beats = segment_beats(filt, peaks; fs=fs)
beats_norm = normalize_beat.(beats)
spiketrain, siglen, _ = get_spiketrain(PATIENT, SESSION; Δ=Δ)

n_spikes = length(spiketrain)
n_up = count(s -> s.polarity, spiketrain)
n_down = count(s -> !s.polarity, spiketrain)
up_ratio = n_up / n_spikes

println("---- Results: $(PATIENT)/$(SESSION) ----")
println("Signal length: $(siglen) samples ($(round(siglen/1000.0, digits=1))s)")
println("Total spikes: $(n_spikes)")
println("Up spikes: $(n_up)  ($(round(100*up_ratio, digits=1))%)")
println("Down spikes: $(n_down)  ($(round(100*(1-up_ratio), digits=1))%)")
println("Spike rate: $(round(n_spikes / (siglen/1000.0), digits=1)) spikes/s")

println("\nFirst 10 spikes:")
for s in spiketrain[1:min(10, end)]
    println("   t=$(round(s.time, digits=1))  polarity=$(s.polarity ? '↑' : '↓')")
end


# --- Plotting ---

t_start, t_end = 1, min(5000, length(raw))  # first 5 seconds
t_axis = (t_start:t_end) ./ fs

# 1. Raw vs filtered
p1 = plot(t_axis, raw[t_start:t_end],
    label="Raw", color=:lightgray, linewidth=1,
    title="Raw vs Filtered",
    xlabel="Time (s)", ylabel="Amplitude")
plot!(p1, t_axis, filt[t_start:t_end],
    label="Filtered", color=:blue, linewidth=1.5)

# R-peaks 
peaks_in_window = filter(p -> t_start ≤ p ≤ t_end, peaks)
scatter!(p1, peaks_in_window ./ fs, filt[peaks_in_window],
    label="R-peaks", color=:red, markersize=5, markershape=:circle)

# 2. Overlay of first 5 normalized beats
p2 = plot(title="Normalized Beat Overlay",
    ylabel="Amplitude", legend=false)
pre = round(Int, 0.25 * fs)
for beat in beats_norm[1:min(5, end)]
    plot!(p2, (-pre:length(beat)-pre-1), beat, alpha=0.7, linewidth=1.5)
end
vline!(p2, [0], color=:red, line=:dash, label="R-peak")

# 3. Spike train raster
up_times   = [s.time for s in spiketrain if s.polarity]  ./ fs
down_times = [s.time for s in spiketrain if !s.polarity] ./ fs

p3 = scatter(up_times,   ones(length(up_times)),
    label="↑", color=:green, markersize=2, markershape=:vline,
    title="Spike Train",
    xlabel="Time (s)", ylabel="", yticks=false)
scatter!(p3, down_times, zeros(length(down_times)),
    label="↓", color=:red, markersize=2, markershape=:vline)

p = plot(p1, p2, p3, layout=(3, 1), size=(1000, 800))
display(p)
savefig(p, joinpath(@__DIR__, "../docs/imgs/signal_pipeline.png"))