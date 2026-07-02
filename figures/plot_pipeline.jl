using Plots

include("../modules/Signals.jl")

using .Signals

# --- Configuration ---
fs = 1000.0
patient, session = "104", "s0306lre" 
out_dir = "./"
mkpath(out_dir)

# Theme settings 
default(
    grid=false, legend=false, titlefontsize=12,
    guidefontsize=10, tickfontsize=8, framestyle=:box,
    size=(400, 250), margin=3Plots.mm
)

# 1. Raw Signal (Macro view: 3 seconds)
raw_sig = load_raw_signal(patient, session)
time_window = 1:round(Int, 3 * fs) 
t_macro = time_window ./ fs
raw_snip = raw_sig[time_window]

p1 = plot(t_macro, raw_snip, title="1. Raw ECG", xlabel="Time (s)", ylabel="Amplitude", color=:gray)
savefig(p1, joinpath(out_dir, "step1_raw.pdf"))

# 2. Filtered Signal
filt_sig = get_filtered_signal(raw_sig; fs=fs)
filt_snip = filt_sig[time_window]

p2 = plot(t_macro, filt_snip, title="2. Filtered Signal", xlabel="Time (s)", ylabel="Amplitude", color=:black)
savefig(p2, joinpath(out_dir, "step2_filtered.pdf"))

# 3. R-Peak Detection
peaks = get_R_peaks(filt_sig; fs=fs)
peaks_in_window = filter(p -> p in time_window, peaks)

p3 = plot(t_macro, filt_snip, title="3. R-Peak Detection", xlabel="Time (s)", ylabel="Amplitude", color=:black)
scatter!(p3, peaks_in_window ./ fs, filt_sig[peaks_in_window], color=:red, markershape=:circle, markersize=5)
savefig(p3, joinpath(out_dir, "step3_peaks.pdf"))

# 4. Segmented Beat (Micro view: 1 beat)
test_peak = peaks_in_window[2] 
beats = segment_beats(filt_sig, [test_peak]; fs=fs)
beat = beats[1]
t_micro = range(-0.25, 0.45, length=length(beat))

p4 = plot(t_micro, beat, title="4. Segmented Beat", xlabel="Time (s)", ylabel="Amplitude", color=:teal, linewidth=2)
savefig(p4, joinpath(out_dir, "step4_segmented.pdf"))

# 5. Normalized Beat
beat_norm = normalize_beat(beat)

p5 = plot(t_micro, beat_norm, title="5. Normalized Beat", xlabel="Time (s)", ylabel="Norm. Amplitude", color=:purple, linewidth=2)
savefig(p5, joinpath(out_dir, "step5_normalized.pdf"))

# 6. Delta Modulation (Spikes)
delta_thresh = 0.1
spikes = delta_modulation(beat_norm; Δ=delta_thresh)

p6 = plot(t_micro, beat_norm, title="6. Delta Modulated Spikes", xlabel="Time (s)", ylabel="Norm. Amplitude", color=:lightgray, alpha=0.6, linewidth=2)

# Extract indices and map them back to the micro time array
up_times = [t_micro[Int(s.time)] for s in spikes if s.polarity == true]
up_y     = [beat_norm[Int(s.time)] for s in spikes if s.polarity == true]

down_times = [t_micro[Int(s.time)] for s in spikes if s.polarity == false]
down_y     = [beat_norm[Int(s.time)] for s in spikes if s.polarity == false]

scatter!(p6, up_times, up_y, color=:blue, markershape=:utriangle, markersize=6, label="UP")
scatter!(p6, down_times, down_y, color=:red, markershape=:dtriangle, markersize=6, label="DOWN")
savefig(p6, joinpath(out_dir, "step6_spikes.pdf"))

println("Successfully generated 6 pipeline plots in $(out_dir)/")