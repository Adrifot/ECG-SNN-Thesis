using DSP
using Plots
using Statistics
gr()


PATIENT = "121"
SESSION = "s0311lre"


struct Spike 
    time::Float64
    polarity::Bool
end


function delta_modulation(signal; Δ=120)
    n = length(signal)
    last_spike_lvl = signal[1]
    spiketrain = Spike[]

    for t in 2:n
        diff = signal[t] - last_spike_lvl
        if diff ≥ Δ
            spike = Spike(t, true)
            push!(spiketrain, spike)
            last_spike_lvl = signal[t]
        elseif diff ≤ -Δ
            spike = Spike(t, false)
            push!(spiketrain, spike)
            last_spike_lvl = signal[t]
        end
    end
    return spiketrain
end


function load_raw_signal(patient, session)
    path = "./ecg-db/patient$(patient)/$(session).dat"
    if !isfile(path)
        error("File not found: $(path)")
    end
    raw_data = reinterpret(Int16, read(path))
    n_channels = 16
    data_matrix = reshape(raw_data, n_channels, :)
    return Float64.(data_matrix[2, :])
end


function get_filtered_signal(signal; lowcut=0.5, highcut=40, fs=1000)
    pass = Bandpass(lowcut, highcut)
    method = Butterworth(4)
    return filtfilt(digitalfilter(pass, method; fs=fs), signal)
end


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
spiketrain = delta_modulation(filtered_signal; Δ=100)

display(plot_spiketrain_demo(filtered_signal, spiketrain))
println("Press Enter to close plot...")
readline()